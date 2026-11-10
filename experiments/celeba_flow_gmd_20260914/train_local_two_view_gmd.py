#!/usr/bin/env python3
"""Isolated CelebA-64 trainer for calibrated local two-view feature GMD.

The experiment warm-starts a two-time flow from the validated GMD-60k EMA
generator and uses the frozen MAE-82 + ConvNeXt-13 representation.  Every
update minimizes

    L = L_boundary(Phi(z,0,1))
        + lambda_local * 0.5 * (L_origin(a,rho) + L_local(a,rho))
        + lambda_e10 * L_eulerian,

where the first term is the released terminal GMD objective and the two local
terms use one detached official-GMD force at ``stopgrad(Phi(z,0,a))``.  The
calibrated hypothesis is explicit: ``force_step=rho`` while physical time is
``rho*(1-a)``.  ``lambda_e10=10`` reproduces the established E10 convention;
it defaults to zero for a pure boundary + local-feature-GMD experiment.

This file does not patch or replace any active trainer.  It requires eight
DDP ranks and the exact 64/128/32 sharded support contract.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import socket
import stat
import sys
import tempfile
import time
import traceback
from typing import Any
import uuid


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_rank_cache() -> None:
    cache_base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
    if not cache_base:
        return
    cache_root = Path(cache_base) / f"rank_{os.environ.get('LOCAL_RANK', '0')}"
    for directory in (cache_root, cache_root / "userdb", cache_root / "kernels"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        MIOPEN_CACHE_DIR=str(cache_root),
        MIOPEN_USER_DB_PATH=str(cache_root / "userdb"),
        MIOPEN_CUSTOM_CACHE_DIR=str(cache_root / "kernels"),
    )


_configure_rank_cache()

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange

import celeba_pixel_gmd_generator_train as base
from celeba_pixel_gmd_mae_train import (
    CelebA64PixelDataset,
    build_dataset_manifest,
    torch_tree_sha256,
)
from experiments.celeba_baseline_20260911 import gmd_train as official
from experiments.celeba_flow_gmd_20260914 import train_composed_gmd as parent_tools
from src.celeba_gmd_composed_rollout import (
    ComposedRollout,
    composed_rollout_artifact_dict,
    flow_from_pinned_parent_payload,
    load_composed_rollout_artifact,
)
from src.celeba_gmd_local_training import (
    DEFAULT_E10_DELTA,
    SUPPORTED_EVALUATION_NFES,
    LocalTwoViewTrainingWrapper,
    TimePairGrid,
    cartesian_time_pair_grid,
    combine_training_losses,
    explicit_time_pair_grid,
    factorized_time_pair_grid,
    global_sharded_boundary_and_local_two_view_loss,
    sample_time_pair,
    unwrap_local_training,
)
from src.celeba_pixel_gmd_convnext import (
    HF_CONFIG_FILENAME,
    HF_CONFIG_SHA256,
    HF_WEIGHTS_FILENAME,
    HF_WEIGHTS_SHA256,
    load_local_convnextv2_base,
)
from src.celeba_pixel_gmd_features import (
    canonical_json_sha256,
    load_frozen_pixel_feature_encoder,
    sha256_file,
    state_dict_sha256,
)
from src.celeba_pixel_gmd_generator import NegativeImageBank, PositiveImageBank
from src.celeba_pixel_gmd_mae_convnext import (
    FrozenMAEConvNextFeatures,
    composite_representation_signature,
)


TRAINING_FORMAT = "driftmodel.celeba64_local_two_view_flow_gmd_training"
TRAINING_VERSION = 1
SIGNATURE_VERSION = 1
PARENT = ROOT / (
    "runs/celeba_baseline_20260911/gmd_sharded_mae82_convnext13_seed42/"
    "checkpoints/checkpoint_step_00060000.pt"
)
PARENT_SHA256 = "177398ff98b38c8ce3e45287a1804bac90243d75d8d7154fc85775ed33f84e8d"
PARENT_STEP = 60_000
PARENT_SEED = 42
PARENT_EMA_SHA256 = "1f0fcdbb6adf9b090ff591366143314af10014e1876f4d9a8de3cbc2aacc3f07"
PARENT_SIGNATURE_SHA256 = "a0463e41b6fc03d9e2f32ffb6c75a8bc0a954d74a312af08493da26e795c0dac"
PARENT_OBJECTIVE = "released_normalized_multitemperature_GMD_MAE82_plus_ConvNeXt13"
TRAIN_FILES_SHA256 = official.TRAIN_FILES_SHA256
CONVNEXT_DIR = ROOT / "data/models/convnextv2-base-22k-224_758ff092"
CONVNEXT_CONFIG = CONVNEXT_DIR / HF_CONFIG_FILENAME
CONVNEXT_WEIGHTS = CONVNEXT_DIR / HF_WEIGHTS_FILENAME

DEFAULT_START_TIMES = (0.1, 0.5, 0.8, 0.9)
DEFAULT_RHOS = (1.0, 0.5, 0.25)
DEFAULT_START_PROBABILITIES = (0.25, 0.25, 0.25, 0.25)
DEFAULT_RHO_PROBABILITIES = (0.2, 0.4, 0.4)
ATTEMPT_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")

SOURCE_FILES = (
    'experiments/celeba_flow_gmd_20260914/train_local_two_view_gmd.py',
    'src/celeba_gmd_local_training.py',
    'src/celeba_gmd_local_two_view.py',
    'src/celeba_gmd_flow_map.py',
    'src/celeba_gmd_flow_objectives.py',
    'src/celeba_gmd_composed_rollout.py',
    'src/celeba_gmd_sharded_features.py',
    'experiments/celeba_flow_gmd_20260914/train_composed_gmd.py',
    'experiments/celeba_baseline_20260911/gmd_train.py',
    'celeba_pixel_gmd_generator_train.py',
    'celeba_pixel_gmd_mae_train.py',
    'src/celeba_pixel_gmd_generator.py',
    'src/celeba_pixel_gmd_features.py',
    'src/official_drifting.py',
    'src/celeba_pixel_gmd_convnext.py',
    'src/celeba_pixel_gmd_mae_convnext.py',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _csv_floats(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated real numbers") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("values must be finite and non-empty")
    return values


def _csv_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    return values


def _csv_pairs(raw: str) -> tuple[tuple[float, float], ...]:
    pairs: list[tuple[float, float]] = []
    try:
        for item in raw.split(","):
            if not item.strip():
                continue
            left, separator, right = item.partition(":")
            if separator != ":" or ":" in right:
                raise ValueError
            pairs.append((float(left.strip()), float(right.strip())))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "pairs must use comma-separated a:rho entries"
        ) from error
    if not pairs:
        raise argparse.ArgumentTypeError("pair list must not be empty")
    return tuple(pairs)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        data_dir=official.TRAIN_DIR,
        train_start_index=0,
        train_end_index=162_770,
        feature_checkpoint=official.MAE_PATH,
        feature_checkpoint_sha256=official.MAE_SHA256,
        expected_feature_training_signature_sha256=official.MAE_SIGNATURE_SHA256,
        feature_weight_kind="ema",
        steps=500,
        checkpoint_every=100,
        log_every=10,
        lr=2e-5,
        lr_warmup_start=1e-6,
        lr_warmup_steps=500,
        global_generated=64,
        positive_count=128,
        positive_bank_size=128,
        negative_count=32,
        negative_bank_size=1_000,
        bank_push=128,
        temperatures=(0.2, 0.05, 0.02),
        output_root=ROOT / "runs/celeba_flow_gmd_20260914/local_training",
        run_label="local_two_view_gmd_seed42",
        verify_gradients=True,
    )
    for action in parser._actions:
        if action.dest in {"feature_checkpoint", "feature_checkpoint_sha256"}:
            action.required = False
    parser.add_argument("--convnext-snapshot-dir", type=Path, default=CONVNEXT_DIR)
    parser.add_argument("--parent-checkpoint", type=Path, default=PARENT)
    parser.add_argument("--parent-checkpoint-sha256", default=PARENT_SHA256)
    parser.add_argument(
        "--time-starts",
        type=_csv_floats,
        default=None,
        help="Cartesian start-time grid; omitted uses 0.1,0.5,0.8,0.9",
    )
    parser.add_argument(
        "--time-rhos",
        type=_csv_floats,
        default=None,
        help="Cartesian rho grid; omitted uses 1,0.5,0.25",
    )
    parser.add_argument(
        "--time-pairs",
        type=_csv_pairs,
        default=None,
        help="explicit ordered a:rho pairs, mutually exclusive with grid flags",
    )
    parser.add_argument(
        "--pair-probabilities",
        type=_csv_floats,
        default=None,
        help="flat ordered pair probabilities summing to one; excludes factor weights",
    )
    parser.add_argument(
        "--start-probabilities",
        type=_csv_floats,
        default=None,
        help="factorized p(a), mutually exclusive with flat pair probabilities",
    )
    parser.add_argument(
        "--rho-probabilities",
        type=_csv_floats,
        default=None,
        help="factorized p(rho), mutually exclusive with flat pair probabilities",
    )
    parser.add_argument("--time-seed", type=int, default=73_001)
    parser.add_argument(
        "--attempt-token",
        default=None,
        help="unique invocation label; autogenerated when omitted and excluded from science signature",
    )
    parser.add_argument(
        "--lambda-local",
        type=float,
        default=0.05,
        help=(
            "local two-view coefficient; zero is a graph-matched terminal "
            "control that still computes every local branch and loss"
        ),
    )
    parser.add_argument(
        "--lambda-e10",
        type=float,
        default=0.0,
        help="coefficient of raw Eulerian MSE; use 10 for the E10 convention",
    )
    parser.add_argument("--e10-delta", type=float, default=DEFAULT_E10_DELTA)
    parser.add_argument("--flow-time-frequency-embedding-size", type=int, default=256)
    parser.add_argument(
        "--sampler-nfes", type=_csv_ints, default=SUPPORTED_EVALUATION_NFES
    )
    parser.add_argument("--variant", default="boundary_plus_local_two_view_feature_gmd")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    invoked = sys.argv[1:] if argv is None else argv
    args.command = shlex.join([sys.argv[0], *invoked])
    return args


def resolve_time_grid(args: argparse.Namespace) -> TimePairGrid:
    if args.time_pairs is not None:
        if (
            args.time_starts is not None
            or args.time_rhos is not None
            or args.start_probabilities is not None
            or args.rho_probabilities is not None
        ):
            raise ValueError("--time-pairs is mutually exclusive with grid/factor flags")
        return explicit_time_pair_grid(args.time_pairs, args.pair_probabilities)
    starts = DEFAULT_START_TIMES if args.time_starts is None else args.time_starts
    rhos = DEFAULT_RHOS if args.time_rhos is None else args.time_rhos
    if args.pair_probabilities is not None:
        if args.start_probabilities is not None or args.rho_probabilities is not None:
            raise ValueError("flat and factorized pair probabilities are mutually exclusive")
        return cartesian_time_pair_grid(starts, rhos, args.pair_probabilities)
    start_probabilities = args.start_probabilities
    rho_probabilities = args.rho_probabilities
    if start_probabilities is None:
        if tuple(starts) == DEFAULT_START_TIMES:
            start_probabilities = DEFAULT_START_PROBABILITIES
        else:
            start_probabilities = tuple(1.0 / len(starts) for _ in starts)
            start_probabilities = (
                *start_probabilities[:-1],
                1.0 - math.fsum(start_probabilities[:-1]),
            )
    if rho_probabilities is None:
        if tuple(rhos) == DEFAULT_RHOS:
            rho_probabilities = DEFAULT_RHO_PROBABILITIES
        else:
            rho_probabilities = tuple(1.0 / len(rhos) for _ in rhos)
            rho_probabilities = (
                *rho_probabilities[:-1],
                1.0 - math.fsum(rho_probabilities[:-1]),
            )
    return factorized_time_pair_grid(
        starts,
        rhos,
        start_probabilities,
        rho_probabilities,
    )


def _source_hashes() -> dict[str, str]:
    missing = [relative for relative in SOURCE_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"required local-training sources are absent: {missing}")
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES}


def verify_source_manifest(args: argparse.Namespace) -> dict[str, str]:
    path = args.expected_source_manifest
    digest = args.expected_source_manifest_sha256
    if path is None or digest is None:
        raise ValueError("training requires a launcher source manifest and SHA-256")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("source manifest must be a regular non-symlink file")
    if not _is_sha256(digest) or sha256_file(path) != digest:
        raise ValueError("source-manifest external SHA-256 differs")
    artifacts = (
        args.feature_checkpoint,
        args.parent_checkpoint,
        CONVNEXT_CONFIG,
        CONVNEXT_WEIGHTS,
    )
    expected_names = [*SOURCE_FILES, *(str(item) for item in artifacts)]
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(expected_names):
        raise ValueError("source/artifact manifest entry count differs")
    parsed: dict[str, str] = {}
    for line, expected_name in zip(lines, expected_names, strict=True):
        observed_digest, separator, observed_name = line.partition("  ")
        if (
            separator != "  "
            or not _is_sha256(observed_digest)
            or observed_name != expected_name
        ):
            raise ValueError(f"source manifest entry differs for {expected_name}")
        parsed[observed_name] = observed_digest
    observed_sources = {relative: parsed[relative] for relative in SOURCE_FILES}
    if observed_sources != _source_hashes():
        raise RuntimeError("training sources changed after launcher hashing")
    expected_artifacts = {
        str(args.feature_checkpoint): official.MAE_SHA256,
        str(args.parent_checkpoint): PARENT_SHA256,
        str(CONVNEXT_CONFIG): HF_CONFIG_SHA256,
        str(CONVNEXT_WEIGHTS): HF_WEIGHTS_SHA256,
    }
    if {name: parsed[name] for name in expected_artifacts} != expected_artifacts:
        raise ValueError("pinned artifact digest differs in source manifest")
    return observed_sources


def validate_objective_weights(lambda_local: float, lambda_e10: float) -> None:
    """Accept finite non-negative weights, including the graph-matched zero control."""

    for value, label in (
        (lambda_local, "lambda_local"),
        (lambda_e10, "lambda_e10"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{label} must be finite and non-negative")


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke:
        raise ValueError("this production trainer has separate CPU unit tests, not --smoke")
    args.time_grid = resolve_time_grid(args)
    if args.world_size != 8 or not args.distributed:
        raise ValueError("local Flow-GMD training requires exactly eight DDP ranks")
    if args.device.type != "cuda" or args.distributed_backend != "nccl":
        raise ValueError("production local Flow-GMD requires ROCm/CUDA with NCCL/RCCL")
    if not args.deterministic:
        raise ValueError("production local Flow-GMD requires deterministic algorithms")
    if args.data_dir.resolve() != official.TRAIN_DIR.resolve():
        raise ValueError("training data directory differs from the pinned split")
    if (args.train_start_index, args.train_end_index) != (0, 162_770):
        raise ValueError("training requires the complete official 162770-image split")
    if (
        args.feature_checkpoint.resolve() != official.MAE_PATH.resolve()
        or args.feature_checkpoint_sha256 != official.MAE_SHA256
        or args.expected_feature_training_signature_sha256
        != official.MAE_SIGNATURE_SHA256
        or args.feature_weight_kind != "ema"
    ):
        raise ValueError("training requires the pinned official MAE-82 EMA")
    if (
        args.parent_checkpoint.resolve() != PARENT.resolve()
        or args.parent_checkpoint_sha256 != PARENT_SHA256
    ):
        raise ValueError("training requires the pinned GMD-60k parent EMA")
    if args.convnext_snapshot_dir.resolve() != CONVNEXT_DIR.resolve():
        raise ValueError("training requires the pinned ConvNeXt-13 snapshot")
    for directory, label in (
        (args.data_dir, "data directory"),
        (args.convnext_snapshot_dir, "ConvNeXt snapshot"),
    ):
        if not directory.is_dir() or directory.is_symlink():
            raise FileNotFoundError(f"{label} must be a regular directory")
    for path, label in (
        (args.feature_checkpoint, "MAE checkpoint"),
        (args.parent_checkpoint, "parent checkpoint"),
        (CONVNEXT_CONFIG, "ConvNeXt config"),
        (CONVNEXT_WEIGHTS, "ConvNeXt weights"),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} must be a regular non-symlink file")
    if args.steps < 1 or args.log_every < 1 or args.checkpoint_every < 0:
        raise ValueError("training/log/checkpoint step counts are invalid")
    if args.global_generated != 64 or args.positive_count != 128:
        raise ValueError("generated/positive supports must remain exactly 64/128")
    if args.negative_count != 32 or args.bank_push != 128:
        raise ValueError("fixed-negative support and bank push must remain 32/128")
    if args.positive_bank_size != 128 or args.negative_bank_size != 1_000:
        raise ValueError("parent-compatible bank capacities must remain 128/1000")
    if any(count % args.world_size for count in (64, 128, 32)):
        raise ValueError("all support counts must divide the DDP world size")
    args.local_generated = 64 // args.world_size
    if tuple(args.temperatures) != (0.2, 0.05, 0.02):
        raise ValueError("official GMD temperatures must remain 0.2,0.05,0.02")
    if args.augmentation != "aligned_flip" or not args.preload_uint8:
        raise ValueError("training requires aligned flip and uint8 preload")
    if not args.use_bfloat16:
        raise ValueError("training requires BF16 generator/MAE autocast")
    expected_architecture = {
        "patch_size": 4,
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "noise_classes": 64,
        "noise_coords": 32,
        "condition_tokens": 16,
        "gradient_checkpointing": True,
    }
    changed = {
        name: {"expected": expected, "observed": getattr(args, name)}
        for name, expected in expected_architecture.items()
        if getattr(args, name) != expected
    }
    if changed:
        raise ValueError(f"generator architecture differs from parent: {changed}")
    if args.flow_time_frequency_embedding_size != 256:
        raise ValueError("the calibrated flow-time embedding size is exactly 256")
    if args.time_seed < 0:
        raise ValueError("time seed must be non-negative")
    validate_objective_weights(args.lambda_local, args.lambda_e10)
    for value, label, allow_zero in (
        (args.e10_delta, "e10_delta", False),
        (args.lr, "learning rate", False),
        (args.grad_clip, "gradient clip", False),
        (args.adam_eps, "Adam epsilon", False),
    ):
        if (
            not math.isfinite(float(value))
            or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
        ):
            raise ValueError(f"{label} has an invalid value")
    if args.e10_delta >= 1.0:
        raise ValueError("e10_delta must be smaller than one")
    if args.lambda_e10 > 0.0 and any(
        a + args.e10_delta >= 1.0
        for (a, rho), probability in zip(
            args.time_grid.pairs, args.time_grid.probabilities, strict=True
        )
        if probability > 0.0
    ):
        raise ValueError("every sampled a must satisfy a+e10_delta<1 when E10 is active")
    if args.lr_warmup_steps < 0 or not 0.0 <= args.lr_warmup_start <= args.lr:
        raise ValueError("learning-rate warmup is invalid")
    if not 0.0 < args.adam_beta1 < 1.0 or not 0.0 < args.adam_beta2 < 1.0:
        raise ValueError("Adam betas must lie strictly between zero and one")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise ValueError("weight decay must be finite and non-negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("EMA decay must lie strictly between zero and one")
    if tuple(args.sampler_nfes) != SUPPORTED_EVALUATION_NFES:
        raise ValueError("one artifact must retain sampler NFE support 1,2,4,8,16")
    if args.resume is None:
        if args.resume_sha256 is not None:
            raise ValueError("--resume-sha256 is valid only with --resume")
    elif (
        not args.resume.is_file()
        or args.resume.is_symlink()
        or not _is_sha256(args.resume_sha256)
    ):
        raise ValueError("resume requires a regular checkpoint and lowercase SHA-256")
    args.launcher_source_hashes = verify_source_manifest(args)


def setup_distributed(args: argparse.Namespace) -> None:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("local Flow-GMD training requires ROCm GPUs")
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=args.distributed_timeout_minutes)
    )
    synchronize_attempt_token(args)
    validate_args(args)


def cleanup(args: argparse.Namespace) -> None:
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def barrier(args: argparse.Namespace) -> None:
    if args.distributed:
        dist.barrier(device_ids=[args.local_rank])


def rank_zero(args: argparse.Namespace, operation):
    barrier(args)
    result = None
    failure = None
    if args.rank == 0:
        try:
            result = operation()
        except Exception:
            failure = traceback.format_exc()
    payload = [failure, result]
    if args.distributed:
        dist.broadcast_object_list(payload, src=0)
        failure, result = payload
    if failure is not None:
        raise RuntimeError(f"rank-zero operation failed:\n{failure}")
    barrier(args)
    return result


def gather(local: Any, args: argparse.Namespace) -> list[Any]:
    if not args.distributed:
        return [local]
    values: list[Any] = [None] * args.world_size
    dist.all_gather_object(values, local)
    return values


def synchronize_attempt_token(args: argparse.Namespace) -> str:
    """Give every rank one invocation-unique, filename-safe audit token."""

    supplied_by_rank = gather(getattr(args, "attempt_token", None), args)
    if any(value != supplied_by_rank[0] for value in supplied_by_rank[1:]):
        raise ValueError("attempt token differs across DDP ranks")
    token = supplied_by_rank[0]
    if token is None:
        token = rank_zero(
            args,
            lambda: (
                "auto_"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}_"
                f"{uuid.uuid4().hex}"
            ),
        )
    if type(token) is not str or ATTEMPT_TOKEN_PATTERN.fullmatch(token) is None:
        raise ValueError(
            "attempt token must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}"
        )
    args.attempt_token = token
    return token


def choose_time_pair(
    grid: TimePairGrid,
    generator: torch.Generator,
    args: argparse.Namespace,
) -> tuple[int, float, float]:
    local_index, local_a, local_rho = sample_time_pair(grid, generator)
    selected = torch.tensor(local_index, device=args.device, dtype=torch.int64)
    if args.distributed:
        dist.broadcast(selected, src=0)
    if int(selected) != local_index:
        raise RuntimeError("time-pair RNG streams diverged across DDP ranks")
    a, rho = grid.pairs[int(selected)]
    if (a, rho) != (local_a, local_rho):
        raise RuntimeError("sampled time-pair index is internally inconsistent")
    return int(selected), a, rho


def _tree_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, Mapping):
        return {key: _tree_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree_cpu(item) for item in value)
    if isinstance(value, list):
        return [_tree_cpu(item) for item in value]
    return copy.deepcopy(value)


def capture_runtime_rng(device: torch.device) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state().cpu(),
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            # Do not pass torch.uint32 tensors through all_gather_object.
            # PyTorch 2.9.1+ROCm 6.3 serializes that dtype through
            # UntypedStorage, then fails while unpickling because the storage
            # class has no ``dtype`` attribute.  Plain Python integers are
            # portable through the object collective and weights-only
            # checkpoint loader, while preserving every MT19937 key exactly.
            "keys": [int(value) for value in numpy_state[1]],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_device": None,
        "device_type": device.type,
    }
    if device.type == "cuda":
        state["torch_device"] = torch.cuda.get_rng_state(device).cpu()
    return state


def restore_runtime_rng(state: Mapping[str, Any], device: torch.device) -> None:
    expected = {"torch_cpu", "python", "numpy", "torch_device", "device_type"}
    if not isinstance(state, Mapping) or set(state) != expected:
        raise ValueError("runtime RNG state schema differs")
    if state["device_type"] != device.type:
        raise ValueError("runtime RNG device type differs")
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping) or set(numpy_state) != {
        "bit_generator",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("NumPy RNG state schema differs")
    numpy_keys = numpy_state["keys"]
    if (
        not isinstance(numpy_keys, list)
        or len(numpy_keys) != 624
        or any(
            type(value) is not int or not 0 <= value <= np.iinfo(np.uint32).max
            for value in numpy_keys
        )
    ):
        raise ValueError("NumPy RNG key state is malformed")
    torch.set_rng_state(state["torch_cpu"].cpu())
    random.setstate(state["python"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_keys, dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    if device.type == "cuda":
        if not isinstance(state["torch_device"], Tensor):
            raise ValueError("CUDA RNG state is absent")
        torch.cuda.set_rng_state(state["torch_device"].cpu(), device)
    elif state["torch_device"] is not None:
        raise ValueError("CPU runtime RNG unexpectedly contains a device state")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_exclusive_text(path: Path, text: str) -> Path:
    """Create and fsync one file, failing atomically if its name exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return path


def write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> Path:
    encoded = json.dumps(
        dict(payload), sort_keys=True, indent=2, ensure_ascii=True
    ) + "\n"
    return write_exclusive_text(path, encoded)


def exclusive_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish a torch payload with a no-replace hard link and SHA sidecar."""

    sidecar = Path(f"{path}.sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path} or its SHA sidecar")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite checkpoint {path}") from error
        _fsync_directory(path.parent)
        digest = sha256_file(path)
        try:
            write_exclusive_text(sidecar, f"{digest}  {path.name}\n")
        except Exception:
            # Keep the already published checkpoint for forensic recovery; it
            # will not be accepted for resume without its exact sidecar.
            raise
        return digest
    finally:
        temporary.unlink(missing_ok=True)


class ExclusiveJsonlLogger:
    """A per-invocation append stream whose path itself is never overwritten."""

    def __init__(self, path: Path) -> None:
        self.path = write_exclusive_text(path, "")

    def log(self, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        payload.setdefault("timestamp_utc", utc_now())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _checkpoint_integrity(payload: Mapping[str, Any]) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "model_state_sha256": state_dict_sha256(payload["model"]),
        "ema_state_sha256": state_dict_sha256(payload["ema"]),
        "optimizer_state_sha256": torch_tree_sha256(payload["optimizer"]),
        "stream_state_sha256": torch_tree_sha256(payload["data_stream"]),
        "positive_bank_state_sha256": torch_tree_sha256(payload["positive_bank"]),
        "negative_bank_state_sha256": torch_tree_sha256(payload["negative_bank"]),
        "bank_rng_state_sha256": torch_tree_sha256(payload["bank_rng_state"]),
        "prior_rng_states_sha256": torch_tree_sha256(payload["prior_rng_by_rank"]),
        "time_rng_state_sha256": torch_tree_sha256(payload["time_rng_state"]),
        "runtime_rng_states_sha256": torch_tree_sha256(payload["runtime_rng_by_rank"]),
        "pair_counts_sha256": torch_tree_sha256(payload["pair_counts"]),
        "training_signature_sha256": canonical_json_sha256(
            dict(payload["training_signature"])
        ),
    }


CHECKPOINT_FIELDS = {
    "format",
    "version",
    "completed_step",
    "model",
    "ema",
    "optimizer",
    "data_stream",
    "positive_bank",
    "negative_bank",
    "bank_rng_state",
    "prior_rng_by_rank",
    "time_rng_state",
    "runtime_rng_by_rank",
    "pair_counts",
    "training_signature",
    "run_dir",
    "saved_at_utc",
    "integrity",
}


def validated_pair_counts(
    counts: Mapping[str, Any],
    *,
    expected_keys: set[str],
    completed_step: int,
) -> dict[str, int]:
    """Validate the exact one-categorical-draw-per-update accounting."""

    if not isinstance(counts, Mapping) or set(counts) != expected_keys:
        raise ValueError("checkpoint pair-count keys differ")
    if type(completed_step) is not int or completed_step < 0:
        raise ValueError("completed step is malformed")
    canonical: dict[str, int] = {}
    for key, value in counts.items():
        if type(key) is not str or type(value) is not int or value < 0:
            raise ValueError("pair counts must be non-negative Python integers")
        canonical[key] = value
    if sum(canonical.values()) != completed_step:
        raise ValueError("pair-count sum must equal completed local-training step")
    return canonical


def save_checkpoint(
    path: Path,
    *,
    completed_step: int,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    stream,
    positive_bank: PositiveImageBank,
    negative_bank: NegativeImageBank,
    bank_generator: torch.Generator,
    prior_generator: torch.Generator,
    time_generator: torch.Generator,
    pair_counts: Mapping[str, int],
    signature: Mapping[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> str:
    wrapper = unwrap_local_training(model)
    canonical_pair_counts = validated_pair_counts(
        pair_counts,
        expected_keys={
            str(index) for index in range(len(args.time_grid.pairs))
        },
        completed_step=completed_step,
    )
    model_sha_by_rank = gather(state_dict_sha256(wrapper.flow.state_dict()), args)
    ema_sha_by_rank = gather(state_dict_sha256(ema.state_dict()), args)
    optimizer_sha_by_rank = gather(torch_tree_sha256(optimizer.state_dict()), args)
    if (
        len(set(model_sha_by_rank)) != 1
        or len(set(ema_sha_by_rank)) != 1
        or len(set(optimizer_sha_by_rank)) != 1
    ):
        raise RuntimeError("model, EMA, or optimizer state differs across DDP ranks")
    prior_states = gather(prior_generator.get_state().cpu(), args)
    runtime_states = gather(capture_runtime_rng(args.device), args)
    shared_local = {
        "data_stream": stream.state_dict(),
        "positive_bank": positive_bank.state_dict(),
        "negative_bank": negative_bank.state_dict(),
        "bank_rng_state": bank_generator.get_state().cpu(),
        "time_rng_state": time_generator.get_state().cpu(),
        "pair_counts": canonical_pair_counts,
    }
    shared_by_rank = gather(shared_local, args)
    if len({torch_tree_sha256(item) for item in shared_by_rank}) != 1:
        raise RuntimeError("stream, banks, time RNG, or pair counts diverged across ranks")
    shared = shared_by_rank[0]

    def publish() -> str:
        payload = {
            "format": TRAINING_FORMAT,
            "version": TRAINING_VERSION,
            "completed_step": int(completed_step),
            "model": _tree_cpu(wrapper.flow.state_dict()),
            "ema": _tree_cpu(ema.state_dict()),
            "optimizer": _tree_cpu(optimizer.state_dict()),
            "data_stream": _tree_cpu(shared["data_stream"]),
            "positive_bank": _tree_cpu(shared["positive_bank"]),
            "negative_bank": _tree_cpu(shared["negative_bank"]),
            "bank_rng_state": _tree_cpu(shared["bank_rng_state"]),
            "prior_rng_by_rank": _tree_cpu(prior_states),
            "time_rng_state": _tree_cpu(shared["time_rng_state"]),
            "runtime_rng_by_rank": _tree_cpu(runtime_states),
            "pair_counts": dict(shared["pair_counts"]),
            "training_signature": copy.deepcopy(dict(signature)),
            "run_dir": str(run_dir.resolve()),
            "saved_at_utc": utc_now(),
        }
        payload["integrity"] = _checkpoint_integrity(payload)
        return exclusive_torch_save(path, payload)

    return str(rank_zero(args, publish))


def load_checkpoint(
    path: Path,
    expected_sha256: str,
    *,
    model: nn.Module,
    ema: nn.Module,
    optimizer: torch.optim.Optimizer,
    stream,
    positive_bank: PositiveImageBank,
    negative_bank: NegativeImageBank,
    bank_generator: torch.Generator,
    prior_generator: torch.Generator,
    time_generator: torch.Generator,
    pair_counts: dict[str, int],
    signature: Mapping[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> int:
    if not _is_sha256(expected_sha256) or sha256_file(path) != expected_sha256:
        raise ValueError("resume checkpoint external SHA-256 differs")
    sidecar = Path(f"{path}.sha256")
    if (
        not sidecar.is_file()
        or sidecar.is_symlink()
        or sidecar.read_text(encoding="utf-8").strip()
        != f"{expected_sha256}  {path.name}"
    ):
        raise ValueError("resume checkpoint SHA sidecar differs")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or set(payload) != CHECKPOINT_FIELDS:
        raise ValueError("resume checkpoint schema differs")
    if payload["format"] != TRAINING_FORMAT or payload["version"] != TRAINING_VERSION:
        raise ValueError("resume checkpoint format/version differs")
    if dict(payload["integrity"]) != _checkpoint_integrity(payload):
        raise ValueError("resume checkpoint internal integrity differs")
    if canonical_json_sha256(dict(payload["training_signature"])) != (
        canonical_json_sha256(dict(signature))
    ) or dict(payload["training_signature"]) != dict(signature):
        raise ValueError("resume training signature differs")
    if Path(payload["run_dir"]).resolve() != run_dir.resolve():
        raise ValueError("resume run directory differs")
    completed_step = payload["completed_step"]
    if type(completed_step) is not int or completed_step < 0:
        raise ValueError("resume completed step is malformed")
    if len(payload["prior_rng_by_rank"]) != args.world_size or len(
        payload["runtime_rng_by_rank"]
    ) != args.world_size:
        raise ValueError("resume per-rank RNG state count differs")
    loaded_pair_counts = validated_pair_counts(
        payload["pair_counts"],
        expected_keys=set(pair_counts),
        completed_step=completed_step,
    )

    wrapper = unwrap_local_training(model)
    wrapper.flow.load_state_dict(payload["model"], strict=True)
    ema.load_state_dict(payload["ema"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    stream.load_state_dict(payload["data_stream"])
    positive_bank.load_state_dict(payload["positive_bank"])
    negative_bank.load_state_dict(payload["negative_bank"])
    bank_generator.set_state(payload["bank_rng_state"].cpu())
    prior_generator.set_state(payload["prior_rng_by_rank"][args.rank].cpu())
    time_generator.set_state(payload["time_rng_state"].cpu())
    pair_counts.clear()
    pair_counts.update(loaded_pair_counts)
    restore_runtime_rng(payload["runtime_rng_by_rank"][args.rank], args.device)

    if state_dict_sha256(wrapper.flow.state_dict()) != payload["integrity"][
        "model_state_sha256"
    ]:
        raise RuntimeError("model state differs immediately after exact resume")
    if state_dict_sha256(ema.state_dict()) != payload["integrity"]["ema_state_sha256"]:
        raise RuntimeError("EMA state differs immediately after exact resume")
    if torch_tree_sha256(optimizer.state_dict()) != payload["integrity"][
        "optimizer_state_sha256"
    ]:
        raise RuntimeError("optimizer state differs immediately after exact resume")
    return completed_step


def verify_parent(path: Path) -> dict[str, Any]:
    identity = parent_tools.verify_parent(path)
    if identity["file_sha256"] != PARENT_SHA256:
        raise ValueError("parent identity differs")
    # The reused verifier describes its own composed-training policy.  Remove
    # those broad booleans and state this trainer's imports atomically so the
    # provenance cannot simultaneously say that an RNG was and was not used.
    for ambiguous in (
        "parent_optimizer_state_imported",
        "parent_stream_state_imported",
        "parent_banks_imported",
        "parent_rng_state_imported",
    ):
        identity.pop(ambiguous, None)
    return {
        **identity,
        "parent_optimizer_state_imported": False,
        "parent_data_stream_state_imported": True,
        "parent_positive_negative_bank_states_imported": True,
        "parent_bank_rng_state_imported": True,
        "parent_prior_rng_states_imported": True,
        "parent_global_runtime_rng_states_imported": False,
    }


def load_parent_payload(args: argparse.Namespace) -> Mapping[str, Any]:
    payload = torch.load(
        args.parent_checkpoint, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(payload, Mapping):
        raise ValueError("parent checkpoint payload is malformed")
    return payload


def load_initial_flow(payload: Mapping[str, Any], args: argparse.Namespace):
    return flow_from_pinned_parent_payload(
        payload,
        expected_step=PARENT_STEP,
        expected_objective=PARENT_OBJECTIVE,
        expected_ema_state_sha256=PARENT_EMA_SHA256,
        expected_training_signature_sha256=PARENT_SIGNATURE_SHA256,
        flow_time_frequency_embedding_size=args.flow_time_frequency_embedding_size,
    )


def import_parent_streams(
    payload: Mapping[str, Any],
    *,
    stream,
    positive_bank: PositiveImageBank,
    negative_bank: NegativeImageBank,
    bank_generator: torch.Generator,
    prior_generator: torch.Generator,
    args: argparse.Namespace,
) -> dict[str, Any]:
    required = {
        "data_stream",
        "positive_bank",
        "negative_bank",
        "bank_rng_state",
        "prior_rng_by_rank",
    }
    if not required.issubset(payload):
        raise ValueError("parent checkpoint lacks stream/bank/RNG fields")
    if len(payload["prior_rng_by_rank"]) != args.world_size:
        raise ValueError("parent prior RNG rank count differs")
    stream.load_state_dict(payload["data_stream"])
    positive_bank.load_state_dict(payload["positive_bank"])
    negative_bank.load_state_dict(payload["negative_bank"])
    bank_generator.set_state(payload["bank_rng_state"].cpu())
    prior_generator.set_state(payload["prior_rng_by_rank"][args.rank].cpu())
    shared_local = {
        "data_stream_sha256": torch_tree_sha256(stream.state_dict()),
        "positive_bank_sha256": torch_tree_sha256(positive_bank.state_dict()),
        "negative_bank_sha256": torch_tree_sha256(negative_bank.state_dict()),
        "bank_rng_sha256": torch_tree_sha256(bank_generator.get_state().cpu()),
    }
    shared_by_rank = gather(shared_local, args)
    if len({canonical_json_sha256(item) for item in shared_by_rank}) != 1:
        raise RuntimeError("imported parent stream, banks, or bank RNG differ by rank")
    prior_hashes = gather(
        torch_tree_sha256(prior_generator.get_state().cpu()), args
    )
    # Return one identical signature fragment on every rank.  The ordered list
    # still records each rank's deliberately distinct parent prior stream.
    return {
        **shared_by_rank[0],
        "prior_rng_sha256_by_rank": prior_hashes,
        "parent_training_seed": PARENT_SEED,
        "data_stream_seed": PARENT_SEED + 101,
    }


def training_signature(
    args: argparse.Namespace,
    dataset_manifest: Mapping[str, Any],
    flow,
    feature_identity: Mapping[str, Any],
    feature_encoder: FrozenMAEConvNextFeatures,
    parent_identity: Mapping[str, Any],
    parent_stream_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "version": SIGNATURE_VERSION,
        "variant": args.variant,
        "objective": "terminal_official_GMD_plus_calibrated_local_two_view_feature_GMD",
        "objective_equation": "L_boundary + lambda_local*0.5*(L_origin+L_local) + lambda_e10*L_eulerian",
        "boundary_implementation": "src.celeba_pixel_gmd_mae_convnext.drifting_loss_from_complete_features",
        "local_implementation": "src.celeba_gmd_local_two_view.two_view_local_feature_gmd_loss",
        "shared_feature_adapter": "src.celeba_gmd_local_training.global_sharded_boundary_and_local_two_view_loss",
        "flow": flow.flow_architecture_config(),
        "initialization": {
            **dict(parent_identity),
            "new_run_seed": int(args.seed),
            "initial_flow_state_sha256": state_dict_sha256(flow.state_dict()),
        },
        "parent_stream_initialization": dict(parent_stream_identity),
        "dataset": dict(dataset_manifest),
        "feature_checkpoint": dict(feature_identity),
        "feature_representation": composite_representation_signature(feature_encoder),
        "support": {
            "generated_global": 64,
            "generated_local_per_rank": args.local_generated,
            "positive": 128,
            "fixed_negative": 32,
            "positive_bank": 128,
            "negative_bank": 1_000,
            "bank_push": 128,
            "fixed_negative_post_affinity_weight": 0.0,
            "positive_negative_features_encoded_once_per_update": True,
            "detached_source_features_and_teacher_reused": True,
        },
        "time_pairs": {
            **args.time_grid.as_dict(),
            "seed": args.time_seed,
            "default_main_factorization": {
                "start_times": list(DEFAULT_START_TIMES),
                "start_probabilities": list(DEFAULT_START_PROBABILITIES),
                "rhos": list(DEFAULT_RHOS),
                "rho_probabilities": list(DEFAULT_RHO_PROBABILITIES),
            },
            "synchronized_across_ranks": True,
            "end_time": "a + rho*(1-a)",
            "physical_delta_time": "rho*(1-a)",
            "force_step": "rho",
            "force_step_hypothesis": "dimensionless GMD pseudo-step, not physical delta time",
            "same_pair_for_complete_global_support": True,
        },
        "loss_weights": {
            "boundary": 1.0,
            "lambda_local": args.lambda_local,
            "lambda_e10": args.lambda_e10,
            "e10_delta": args.e10_delta,
            "e10_convention": "lambda_e10=10 is E10; zero disables its extra calls",
            "lambda_local_zero_control": (
                "graph-matched: source/origin/local forwards, feature extraction, "
                "teacher and local loss remain computed; only weighted gradient is zero"
            ),
        },
        "local_semigradient": {
            "source": "stopgrad(Phi(z,0,a))",
            "official_GMD_force": "detached",
            "student_views": ["Phi(z,0,u)", "Phi(stopgrad(Phi(z,0,a)),a,u)"],
            "view_weighting": "exact arithmetic half-sum",
            "same_noise_labels": True,
            "noise_reinjection": False,
        },
        "temperatures": list(args.temperatures),
        "optimizer": {
            "name": "AdamW",
            "fresh_at_local_step_zero": True,
            "parent_optimizer_imported": False,
            "lr": args.lr,
            "warmup_start": args.lr_warmup_start,
            "warmup_steps": args.lr_warmup_steps,
            "betas": [args.adam_beta1, args.adam_beta2],
            "eps": args.adam_eps,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.grad_clip,
        },
        "ema": {"decay": args.ema_decay, "initialized_from": "flow_theta_0"},
        "precision": {
            "parameters_optimizer_and_GMD": "float32",
            "generator_and_MAE": "bfloat16_autocast",
            "ConvNeXt": "float32",
        },
        "distributed": {
            "world_size": 8,
            "sole_trainable_DDP_boundary": "LocalTwoViewTrainingWrapper",
            "all_flow_calls_inside_wrapper_forward": True,
            "feature_support_sharding": True,
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "static_graph": True,
        },
        "sampler": {
            "supported_nfes": list(SUPPORTED_EVALUATION_NFES),
            "fixed_labels_across_steps": True,
            "one_artifact_all_nfes": True,
            "terminal_clip": False,
        },
        "source_files_sha256": dict(args.launcher_source_hashes),
        "scientific_claim_limit": (
            "force_step=rho is a calibrated pilot hypothesis; quality and "
            "monotone FID-vs-NFE require held-out evaluation"
        ),
    }


def resolve_run_dir(args: argparse.Namespace) -> Path:
    def resolve() -> str:
        if args.resume is not None:
            parent = args.resume.resolve().parent
            inferred = parent.parent if parent.name == "checkpoints" else parent
            if args.run_dir is not None and args.run_dir.resolve() != inferred:
                raise ValueError("--run-dir differs from directory implied by --resume")
            if not inferred.is_dir() or inferred.is_symlink():
                raise FileNotFoundError("resume run directory is absent or a symlink")
            return str(inferred)
        if args.run_dir is not None:
            path = args.run_dir.resolve()
        else:
            label = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in args.run_label
            ).strip("_")
            path = (
                args.output_root
                / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{label}"
            ).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
        return str(path)

    return Path(rank_zero(args, resolve))


def reserve_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Hold a kernel-reclaimed run lock and publish immutable owner metadata."""

    def reserve() -> dict[str, Any]:
        lock = run_dir / ".local_two_view_training.lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(lock, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise RuntimeError("run lock must be a regular file")
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("run directory is held by an active invocation") from error
            owners = run_dir / "lock_owners"
            owners.mkdir(exist_ok=True)
            if not owners.is_dir() or owners.is_symlink():
                raise RuntimeError("lock owner directory must be a real directory")
            owner = owners / f"attempt_{args.attempt_token}.json"
            token = {
                "attempt_token": args.attempt_token,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "created_at_utc": utc_now(),
                "lock_recovery": "kernel_flock_released_automatically_on_process_exit",
            }
            write_exclusive_json(owner, token)
            # The descriptor intentionally remains open on rank zero for the
            # full training invocation.  flock is released by the kernel even
            # after SIGKILL or node/process failure.
            return {
                "lock": str(lock),
                "owner": str(owner),
                "descriptor": descriptor,
            }
        except Exception:
            os.close(descriptor)
            raise

    return rank_zero(args, reserve)


def release_run(reservation: Mapping[str, Any] | None, args: argparse.Namespace) -> None:
    if reservation is None:
        return

    def release() -> None:
        lock = Path(reservation["lock"])
        owner = Path(reservation["owner"])
        descriptor = reservation["descriptor"]
        if (
            type(descriptor) is not int
            or owner
            != lock.parent / "lock_owners" / f"attempt_{args.attempt_token}.json"
            or not owner.is_file()
            or owner.is_symlink()
            or not lock.is_file()
            or lock.is_symlink()
        ):
            raise RuntimeError("run reservation ownership differs")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    rank_zero(args, release)


def _snapshot_sources(
    run_dir: Path,
    expected: Mapping[str, str],
    name: str,
) -> dict[str, Any]:
    destination = run_dir / name
    destination.mkdir(exist_ok=False)
    before = _source_hashes()
    if before != dict(expected):
        raise RuntimeError("source snapshot differs from launcher manifest")
    for relative in SOURCE_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    copied = {relative: sha256_file(destination / relative) for relative in SOURCE_FILES}
    if copied != before or _source_hashes() != before:
        raise RuntimeError("sources changed while snapshotting")
    manifest = {
        "format": "driftmodel.celeba64_local_two_view_source_snapshot",
        "version": 1,
        "created_at_utc": utc_now(),
        "sha256": copied,
    }
    manifest_path = write_exclusive_json(destination / "manifest.json", manifest)
    return {
        **manifest,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }


def write_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    dataset_manifest: Mapping[str, Any],
    signature: Mapping[str, Any],
    flow: nn.Module,
    *,
    start_step: int,
) -> None:
    suffix = (
        f"attempt_{args.attempt_token}_segment_"
        f"{start_step:08d}_to_{args.steps:08d}"
    )
    snapshot = _snapshot_sources(
        run_dir, args.launcher_source_hashes, f"source_snapshot_{suffix}"
    )
    write_exclusive_json(
        run_dir / f"dataset_manifest_{suffix}.json", dataset_manifest
    )
    config = {
        key: (
            value.as_dict()
            if isinstance(value, TimePairGrid)
            else str(value)
            if isinstance(value, (Path, torch.device))
            else value
        )
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    config.update(
        parameter_count=sum(parameter.numel() for parameter in flow.parameters()),
        training_signature=dict(signature),
        training_signature_sha256=canonical_json_sha256(dict(signature)),
    )
    write_exclusive_json(
        run_dir / f"config_{suffix}.json",
        json.loads(json.dumps(config, default=str)),
    )
    write_exclusive_json(
        run_dir / f"provenance_{suffix}.json",
        {
            "created_at_utc": utc_now(),
            "command": args.command,
            "hostname": socket.gethostname(),
            "torch": str(torch.__version__),
            "hip": getattr(torch.version, "hip", None),
            "world_size": args.world_size,
            "source_snapshot": snapshot,
            "parent": signature["initialization"],
        },
    )


def verify_gradients(
    model: nn.Module,
    feature_encoder: FrozenMAEConvNextFeatures,
    outputs,
    args: argparse.Namespace,
) -> None:
    if any(parameter.grad is not None for parameter in feature_encoder.parameters()):
        raise RuntimeError("frozen feature encoder received parameter gradients")
    gradients = [parameter.grad for parameter in model.parameters()]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("a trainable Flow-GMD parameter has no gradient")
    assert all(gradient is not None for gradient in gradients)
    finite = torch.stack([torch.isfinite(gradient).all() for gradient in gradients])
    passed = finite.all() and outputs.source.grad_fn is None and not outputs.source.requires_grad
    status = torch.tensor(int(bool(passed)), device=args.device, dtype=torch.int32)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    if not bool(status):
        raise FloatingPointError("gradient/source-detach invariant failed")
    local_retained = feature_encoder.last_local_generated
    if local_retained is None or local_retained.grad is None:
        raise RuntimeError("two-view RGB gradient was not retained")
    if args.lambda_local == 0.0:
        local_zero = torch.isfinite(local_retained.grad).all() & (
            local_retained.grad.abs().sum() == 0
        )
        terminal_retained = outputs.terminal
        if terminal_retained.grad is None:
            raise RuntimeError("terminal RGB gradient was not retained for zero-local control")
        rgb_valid = (
            local_zero
            & torch.isfinite(terminal_retained.grad).all()
            & (terminal_retained.grad.abs().sum() > 0)
        )
    else:
        rgb_valid = torch.isfinite(local_retained.grad).all() & (
            local_retained.grad.abs().sum() > 0
        )
    rgb_status = rgb_valid.to(dtype=torch.int32)
    dist.all_reduce(rgb_status, op=dist.ReduceOp.MIN)
    if not bool(rgb_status):
        raise FloatingPointError(
            "weighted RGB-gradient invariant failed for the selected local coefficient"
        )


def _compact_local_statistics(statistics: Mapping[str, Tensor]) -> dict[str, Tensor]:
    family_origin = [
        value for key, value in statistics.items() if key.startswith("loss/origin/")
    ]
    family_local = [
        value for key, value in statistics.items() if key.startswith("loss/local/")
    ]
    force_rms = [
        value
        for key, value in statistics.items()
        if key.startswith("force/rms_global/")
    ]
    if not family_origin or len(family_origin) != len(family_local) or not force_rms:
        raise RuntimeError("local feature-GMD statistics are incomplete")
    return {
        "local_two_view_gmd_loss": statistics["loss"],
        "local_origin_gmd_loss": statistics["loss/origin"],
        "local_local_gmd_loss": statistics["loss/local"],
        "local_origin_family_mean": torch.stack(family_origin).mean(),
        "local_local_family_mean": torch.stack(family_local).mean(),
        "local_force_rms_family_mean": torch.stack(force_rms).mean(),
        "local_feature_family_count": float(len(family_origin)),
        "local_force_step": statistics["force_step"],
    }


def ensure_sha_sidecar(path: Path, digest: str) -> Path:
    """Create a missing SHA sidecar, or strictly verify the existing one."""

    if not _is_sha256(digest):
        raise ValueError("artifact digest must be a lowercase SHA-256")
    sidecar = Path(f"{path}.sha256")
    expected = f"{digest}  {path.name}"
    if sidecar.is_symlink():
        raise ValueError(f"SHA sidecar must not be a symlink: {sidecar}")
    if sidecar.exists():
        if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != expected:
            raise ValueError(f"SHA sidecar differs: {sidecar}")
    else:
        write_exclusive_text(sidecar, f"{expected}\n")
    return sidecar


def publish_or_verify_flow_artifact(
    path: Path,
    flow: nn.Module,
    metadata: Mapping[str, Any],
) -> str:
    """Publish once, or verify an interrupted export without overwriting it."""

    sidecar = Path(f"{path}.sha256")
    if path.is_symlink() or sidecar.is_symlink():
        raise ValueError("flow artifact and sidecar must not be symlinks")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"flow artifact is not a regular file: {path}")
        digest = sha256_file(path)
    else:
        if sidecar.exists():
            raise ValueError("artifact sidecar exists without its artifact")
        digest = exclusive_torch_save(
            path,
            composed_rollout_artifact_dict(
                ComposedRollout(flow), metadata=metadata
            ),
        )

    loaded, observed_metadata = load_composed_rollout_artifact(path, "cpu")
    expected_metadata = copy.deepcopy(dict(metadata))
    observed_metadata = dict(observed_metadata)
    expected_created = expected_metadata.pop("created_at_utc", None)
    observed_created = observed_metadata.pop("created_at_utc", None)
    if (
        type(expected_created) is not str
        or type(observed_created) is not str
        or canonical_json_sha256(observed_metadata)
        != canonical_json_sha256(expected_metadata)
        or observed_metadata != expected_metadata
    ):
        raise ValueError(f"existing flow artifact metadata differs: {path}")
    if state_dict_sha256(loaded.flow.state_dict()) != state_dict_sha256(
        flow.state_dict()
    ):
        raise ValueError(f"existing flow artifact state differs: {path}")
    del loaded
    ensure_sha_sidecar(path, digest)
    return digest


def publish_or_verify_json(path: Path, payload: Mapping[str, Any]) -> str:
    """Publish deterministic JSON once, or verify it after interrupted export."""

    canonical = json.loads(json.dumps(dict(payload), sort_keys=True))
    sidecar = Path(f"{path}.sha256")
    if path.is_symlink() or sidecar.is_symlink():
        raise ValueError("JSON artifact and sidecar must not be symlinks")
    if path.exists():
        if not path.is_file():
            raise ValueError(f"JSON artifact is not a regular file: {path}")
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"existing JSON artifact is malformed: {path}") from error
        if (
            observed != canonical
            or canonical_json_sha256(observed) != canonical_json_sha256(canonical)
        ):
            raise ValueError(f"existing JSON artifact differs: {path}")
    else:
        if sidecar.exists():
            raise ValueError("JSON sidecar exists without its artifact")
        write_exclusive_json(path, canonical)
    digest = sha256_file(path)
    ensure_sha_sidecar(path, digest)
    return digest


def export_artifacts(
    run_dir: Path,
    completed_step: int,
    flow,
    ema,
    signature: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> None:
    def metadata(weight_kind: str) -> dict[str, Any]:
        return {
            "completed_step": completed_step,
            "weight_kind": weight_kind,
            "training_signature": dict(signature),
            "training_signature_sha256": canonical_json_sha256(dict(signature)),
            "supported_nfes": list(SUPPORTED_EVALUATION_NFES),
            "created_at_utc": utc_now(),
        }

    raw_path = run_dir / f"flow_raw_step_{completed_step:08d}.pt"
    ema_path = run_dir / f"flow_ema_step_{completed_step:08d}.pt"
    raw_digest = publish_or_verify_flow_artifact(
        raw_path,
        flow,
        metadata("raw"),
    )
    ema_digest = publish_or_verify_flow_artifact(
        ema_path,
        ema,
        metadata("ema"),
    )
    manifest = {
        "format": "driftmodel.celeba64_local_two_view_flow_gmd_artifacts",
        "version": 1,
        "completed_step": completed_step,
        "strict_reload": True,
        "supported_nfes": list(SUPPORTED_EVALUATION_NFES),
        "artifacts": {
            "training_checkpoint": {
                "path": str(checkpoint_path.resolve()),
                "sha256": checkpoint_sha256,
            },
            "raw": {"path": str(raw_path.resolve()), "sha256": raw_digest},
            "ema": {"path": str(ema_path.resolve()), "sha256": ema_digest},
        },
    }
    publish_or_verify_json(
        run_dir / f"artifact_manifest_step_{completed_step:08d}.json",
        manifest,
    )


def train(args: argparse.Namespace) -> Path:
    parent_identity = rank_zero(args, lambda: verify_parent(args.parent_checkpoint))
    dataset_manifest = rank_zero(
        args,
        lambda: build_dataset_manifest(
            args.data_dir.resolve(), args.train_start_index, args.train_end_index
        ),
    )
    if (
        dataset_manifest["files_sha256"] != TRAIN_FILES_SHA256
        or dataset_manifest["image_count"] != 162_770
    ):
        raise ValueError("official dataset manifest differs")
    dataset = CelebA64PixelDataset(
        args.data_dir.resolve(), dataset_manifest, preload_uint8=True
    )
    if rank_zero(
        args,
        lambda: build_dataset_manifest(args.data_dir.resolve(), 0, 162_770),
    ) != dataset_manifest:
        raise RuntimeError("dataset changed while preloading")

    # The stream state stores its cursor but not its permutation seed.  It must
    # therefore be reconstructed with the pinned parent's seed, independently
    # of the new run's parameter/RNG seed, before loading the parent cursor.
    stream = base.GlobalBankStream(dataset, batch_size=128, seed=PARENT_SEED + 101)
    positive_bank = PositiveImageBank(args.positive_bank_size)
    negative_bank = NegativeImageBank(args.negative_bank_size)
    bank_generator = base.make_bank_generator(args)
    prior_generator = base.make_prior_generator(args)
    time_generator = torch.Generator(device="cpu").manual_seed(args.time_seed)

    feature_identity = base.feature_checkpoint_identity(args)
    mae = load_frozen_pixel_feature_encoder(args.feature_checkpoint, args.device)
    convnext = load_local_convnextv2_base(
        args.convnext_snapshot_dir, device=args.device
    )
    feature_encoder = FrozenMAEConvNextFeatures(mae, convnext).eval()

    base.seed_parameters(args)
    parent_payload = load_parent_payload(args)
    flow = load_initial_flow(parent_payload, args).to(args.device)
    parent_stream_identity = import_parent_streams(
        parent_payload,
        stream=stream,
        positive_bank=positive_bank,
        negative_bank=negative_bank,
        bank_generator=bank_generator,
        prior_generator=prior_generator,
        args=args,
    )
    del parent_payload
    ema = copy.deepcopy(flow).eval().requires_grad_(False)
    wrapper = LocalTwoViewTrainingWrapper(
        flow,
        enable_e10=args.lambda_e10 > 0.0,
        e10_delta=args.e10_delta,
    ).train()
    model: nn.Module = DistributedDataParallel(
        wrapper,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        static_graph=True,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    if optimizer.state:
        raise RuntimeError("local Flow-GMD optimizer must start empty")
    signature = training_signature(
        args,
        dataset_manifest,
        flow,
        feature_identity,
        feature_encoder,
        parent_identity,
        parent_stream_identity,
    )
    signature_hashes = gather(canonical_json_sha256(signature), args)
    if len(set(signature_hashes)) != 1:
        raise RuntimeError("training signature differs across DDP ranks")
    run_dir = resolve_run_dir(args)
    reservation: Mapping[str, Any] | None = None
    pair_counts = {str(index): 0 for index in range(len(args.time_grid.pairs))}
    try:
        reservation = reserve_run(run_dir, args)
        start_step = 0
        if args.resume is not None:
            start_step = load_checkpoint(
                args.resume,
                args.resume_sha256,
                model=model,
                ema=ema,
                optimizer=optimizer,
                stream=stream,
                positive_bank=positive_bank,
                negative_bank=negative_bank,
                bank_generator=bank_generator,
                prior_generator=prior_generator,
                time_generator=time_generator,
                pair_counts=pair_counts,
                signature=signature,
                run_dir=run_dir,
                args=args,
            )
        if start_step > args.steps:
            raise ValueError("--steps must not precede the resumed local-training step")
        export_only = start_step == args.steps
        rank_zero(
            args,
            lambda: write_metadata(
                args,
                run_dir,
                dataset_manifest,
                signature,
                flow,
                start_step=start_step,
            ),
        )
        logger = None
        if args.rank == 0:
            logger = ExclusiveJsonlLogger(
                run_dir
                / (
                    f"metrics_attempt_{args.attempt_token}_segment_"
                    f"{start_step:08d}_to_{args.steps:08d}.jsonl"
                )
            )
            logger.log(
                {
                    "event": "resume" if start_step else "start",
                    "step": start_step,
                    "world_size": args.world_size,
                    "time_grid": args.time_grid.as_dict(),
                    "lambda_local": args.lambda_local,
                    "lambda_e10": args.lambda_e10,
                    "parent_step": PARENT_STEP,
                    "fresh_optimizer": start_step == 0,
                    "parent_streams_imported": start_step == 0,
                    "export_only": export_only,
                    "parameter_count": sum(p.numel() for p in flow.parameters()),
                    "supported_sampler_nfes": list(SUPPORTED_EVALUATION_NFES),
                }
            )

        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.monotonic()
        progress = trange(
            start_step,
            args.steps,
            initial=start_step,
            total=args.steps,
            desc="celeba64:local-two-view-gmd",
            disable=args.rank != 0,
        )
        last_checkpoint_path: Path | None = args.resume if export_only else None
        last_checkpoint_digest: str | None = (
            args.resume_sha256 if export_only else None
        )
        last_checkpoint_step: int | None = start_step if export_only else None
        for step_index in progress:
            update_started = time.monotonic()
            learning_rate = base.learning_rate_at_step(step_index, args)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            pushed = base.augment_bank_images(
                stream.next(), "aligned_flip", generator=bank_generator
            )
            positive_bank.add(pushed)
            negative_bank.add(pushed)
            positives = positive_bank.sample(
                args.positive_count, generator=bank_generator
            ).to(args.device, non_blocking=True)
            negatives = negative_bank.sample(
                args.negative_count, generator=bank_generator
            ).to(args.device, non_blocking=True)
            prior, labels = unwrap_local_training(model).flow.sample_prior(
                args.local_generated,
                device=args.device,
                generator=prior_generator,
            )
            pair_index, start_time, rho = choose_time_pair(
                args.time_grid, time_generator, args
            )
            pair_counts[str(pair_index)] += 1

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(prior, labels, start_time, rho)
                (
                    boundary_loss,
                    boundary_details,
                    local_loss,
                    local_details,
                ) = global_sharded_boundary_and_local_two_view_loss(
                    feature_encoder,
                    outputs.terminal,
                    outputs.source,
                    outputs.origin_student,
                    outputs.local_student,
                    positives,
                    negatives,
                    force_step=rho,
                    temperatures=args.temperatures,
                )
                total_loss = combine_training_losses(
                    boundary_loss,
                    local_loss,
                    outputs.e10,
                    lambda_local=args.lambda_local,
                    lambda_e10=args.lambda_e10,
                )
            finite = torch.isfinite(total_loss.detach()).to(torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not bool(finite):
                raise FloatingPointError(
                    f"non-finite local Flow-GMD loss at step {step_index + 1}"
                )
            total_loss.backward()
            completed = step_index + 1
            detailed = args.verify_gradients and (
                completed == 1
                or (
                    args.gradient_verification_interval
                    and completed % args.gradient_verification_interval == 0
                )
            )
            if detailed:
                verify_gradients(model, feature_encoder, outputs, args)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            base.update_ema(ema, unwrap_local_training(model).flow, args.ema_decay)
            feature_encoder.clear_step_state()

            if completed == 1 or completed == args.steps or completed % args.log_every == 0:
                torch.cuda.synchronize(args.device)
                reduced = base.reduce_stats(
                    {
                        **{
                            f"boundary_{key}": value
                            for key, value in base.compact_gmd_statistics(
                                boundary_details
                            ).items()
                        },
                        **_compact_local_statistics(local_details),
                        "loss_total": total_loss.detach(),
                        "loss_e10_raw": outputs.e10.detach(),
                        "gradient_norm_preclip": gradient_norm,
                        "terminal_mean": outputs.terminal.detach().float().mean(),
                        "terminal_std": outputs.terminal.detach().float().std(
                            correction=0
                        ),
                    },
                    args,
                )
                maxima = base.reduce_max_stats(
                    {
                        "seconds_per_update_max_rank": time.monotonic()
                        - update_started,
                        "gpu_peak_memory_allocated_bytes_max_rank": torch.cuda.max_memory_allocated(
                            args.device
                        ),
                        "gpu_peak_memory_reserved_bytes_max_rank": torch.cuda.max_memory_reserved(
                            args.device
                        ),
                    },
                    args,
                )
                record = {
                    "event": "train",
                    "step": completed,
                    "pair_index": pair_index,
                    "start_time": start_time,
                    "rho": rho,
                    "end_time": start_time + rho * (1.0 - start_time),
                    "physical_delta_time": rho * (1.0 - start_time),
                    "force_step": rho,
                    "pair_counts": dict(pair_counts),
                    "learning_rate": learning_rate,
                    "lambda_local": args.lambda_local,
                    "lambda_e10": args.lambda_e10,
                    "elapsed_seconds_this_invocation": time.monotonic() - started,
                    "detailed_gradient_check": detailed,
                    **{name: float(value.cpu()) for name, value in reduced.items()},
                    **{name: float(value.cpu()) for name, value in maxima.items()},
                }
                if logger is not None:
                    logger.log(record)
                    progress.set_postfix(
                        boundary=f"{record['boundary_gmd_loss']:.3f}",
                        local=f"{record['local_two_view_gmd_loss']:.3f}",
                    )
            if args.checkpoint_every and completed % args.checkpoint_every == 0:
                checkpoint_path = (
                    run_dir
                    / "checkpoints"
                    / f"checkpoint_step_{completed:08d}.pt"
                )
                checkpoint_digest = save_checkpoint(
                    checkpoint_path,
                    completed_step=completed,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    stream=stream,
                    positive_bank=positive_bank,
                    negative_bank=negative_bank,
                    bank_generator=bank_generator,
                    prior_generator=prior_generator,
                    time_generator=time_generator,
                    pair_counts=pair_counts,
                    signature=signature,
                    run_dir=run_dir,
                    args=args,
                )
                last_checkpoint_path = checkpoint_path
                last_checkpoint_digest = checkpoint_digest
                last_checkpoint_step = completed
                if logger is not None:
                    logger.log(
                        {
                            "event": "checkpoint",
                            "step": completed,
                            "path": str(checkpoint_path.resolve()),
                            "sha256": checkpoint_digest,
                        }
                    )

        if last_checkpoint_step != args.steps:
            last_checkpoint_path = (
                run_dir / "checkpoints" / f"checkpoint_step_{args.steps:08d}.pt"
            )
            last_checkpoint_digest = save_checkpoint(
                last_checkpoint_path,
                completed_step=args.steps,
                model=model,
                ema=ema,
                optimizer=optimizer,
                stream=stream,
                positive_bank=positive_bank,
                negative_bank=negative_bank,
                bank_generator=bank_generator,
                prior_generator=prior_generator,
                time_generator=time_generator,
                pair_counts=pair_counts,
                signature=signature,
                run_dir=run_dir,
                args=args,
            )
            last_checkpoint_step = args.steps
        assert last_checkpoint_path is not None
        assert last_checkpoint_digest is not None
        parent_sha_reverified = rank_zero(
            args, lambda: sha256_file(args.parent_checkpoint)
        )
        if parent_sha_reverified != PARENT_SHA256:
            raise RuntimeError("pinned parent changed during local training")
        rank_zero(
            args,
            lambda: export_artifacts(
                run_dir,
                args.steps,
                unwrap_local_training(model).flow,
                ema,
                signature,
                last_checkpoint_path,
                last_checkpoint_digest,
            ),
        )
        if logger is not None:
            logger.log(
                {
                    "event": "complete",
                    "step": args.steps,
                    "export_only": export_only,
                    "elapsed_seconds_this_invocation": time.monotonic() - started,
                    "final_checkpoint_sha256": last_checkpoint_digest,
                    "pair_counts": dict(pair_counts),
                    "parent_sha256_reverified": True,
                }
            )
        return run_dir
    finally:
        release_run(reservation, args)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    try:
        setup_distributed(args)
        return train(args)
    finally:
        cleanup(args)


if __name__ == "__main__":
    main()
