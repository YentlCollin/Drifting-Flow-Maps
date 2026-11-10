#!/usr/bin/env python3
"""Train the CelebA direct-RGB flow with terminal and structural GMD losses.

Each update makes exactly one call through one outer
``StructuralComposedRollout`` DDP boundary.  That call shares a rank-local
prior and fixed noise labels between a random-depth terminal rollout, a
stopped-teacher local Eulerian composition, and a stopped-teacher canonical
semigroup composition.  The terminal images retain the official globally
sharded MAE82+ConvNeXt13 GMD support contract (64 generated, 128 positive,
32 fixed negative images).

Both structural weights default to zero as a fail-safe.  A branch is built
only when its corresponding CLI weight is explicitly positive.

This is a standalone experimental trainer.  It initializes only the flow
weights from the pinned 60k EMA parent; optimizer, EMA, streams, banks, and
RNGs are fresh unless a strict checkpoint resume is requested.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import socket
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _configure_rank_cache() -> None:
    cache_base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
    if not cache_base:
        return
    root = Path(cache_base) / f"rank_{os.environ.get('LOCAL_RANK', '0')}"
    for directory in (root, root / "userdb", root / "kernels"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        MIOPEN_CACHE_DIR=str(root),
        MIOPEN_USER_DB_PATH=str(root / "userdb"),
        MIOPEN_CUSTOM_CACHE_DIR=str(root / "kernels"),
    )


_configure_rank_cache()

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange

import celeba_pixel_gmd_generator_train as base
from celeba_pixel_gmd_mae_train import (
    CelebA64PixelDataset,
    JsonlLogger,
    atomic_json_dump,
    atomic_text_write,
    atomic_torch_save,
    build_dataset_manifest,
    torch_tree_sha256,
)
from experiments.celeba_baseline_20260911 import gmd_train as official
from src.celeba_gmd_composed_rollout import (
    ComposedRollout,
    composed_rollout_artifact_dict,
    flow_from_pinned_parent_payload,
    load_composed_rollout_artifact,
)
from src.celeba_gmd_sharded_features import global_sharded_mae_convnext_loss
from src.celeba_gmd_structural_rollout import (
    StructuralComposedRollout,
    structural_losses,
    unwrap_structural_composed_rollout,
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


TRAINING_FORMAT = "driftmodel.celeba64_flow_gmd_structural_training"
TRAINING_VERSION = 1
SIGNATURE_VERSION = 1
PARENT = ROOT / (
    "runs/celeba_baseline_20260911/gmd_sharded_mae82_convnext13_seed42/"
    "checkpoints/checkpoint_step_00060000.pt"
)
PARENT_SHA256 = "177398ff98b38c8ce3e45287a1804bac90243d75d8d7154fc85775ed33f84e8d"
PARENT_STEP = 60_000
PARENT_EMA_SHA256 = "1f0fcdbb6adf9b090ff591366143314af10014e1876f4d9a8de3cbc2aacc3f07"
PARENT_SIGNATURE_SHA256 = "a0463e41b6fc03d9e2f32ffb6c75a8bc0a954d74a312af08493da26e795c0dac"
PARENT_OBJECTIVE = "released_normalized_multitemperature_GMD_MAE82_plus_ConvNeXt13"
TRAIN_FILES_SHA256 = official.TRAIN_FILES_SHA256

CONVNEXT_DIR = ROOT / "data/models/convnextv2-base-22k-224_758ff092"
CONVNEXT_CONFIG = CONVNEXT_DIR / HF_CONFIG_FILENAME
CONVNEXT_WEIGHTS = CONVNEXT_DIR / HF_WEIGHTS_FILENAME

SOURCE_FILES = (
    'experiments/celeba_flow_gmd_20260914/train_structural_gmd.py',
    'src/celeba_gmd_structural_rollout.py',
    'src/celeba_gmd_composed_rollout.py',
    'src/celeba_gmd_flow_map.py',
    'src/celeba_gmd_flow_objectives.py',
    'src/celeba_gmd_sharded_features.py',
    'experiments/celeba_baseline_20260911/gmd_train.py',
    'celeba_pixel_gmd_generator_train.py',
    'celeba_pixel_gmd_mae_train.py',
    'src/celeba_pixel_gmd_generator.py',
    'src/celeba_pixel_gmd_features.py',
    'src/official_drifting.py',
    'src/celeba_pixel_gmd_convnext.py',
    'src/celeba_pixel_gmd_mae_convnext.py',
)


def _csv_ints(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "rollout NFEs must be comma-separated integers"
        ) from error
    if not values or any(value < 1 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(
            "rollout NFEs must be distinct positive integers"
        )
    return values


def _csv_floats(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "probabilities must be comma-separated floats"
        ) from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("probabilities must be finite and positive")
    return values


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
        output_root=ROOT / "runs/celeba_flow_gmd_20260914/structural_training",
        run_label="structural_gmd_k124_explicit_lambdas_seed42",
        verify_gradients=True,
    )
    for action in parser._actions:
        if action.dest in {"feature_checkpoint", "feature_checkpoint_sha256"}:
            action.required = False
    parser.add_argument("--convnext-snapshot-dir", type=Path, default=CONVNEXT_DIR)
    parser.add_argument("--parent-checkpoint", type=Path, default=PARENT)
    parser.add_argument("--parent-checkpoint-sha256", default=PARENT_SHA256)
    parser.add_argument("--rollout-nfes", type=_csv_ints, default=(1, 2, 4))
    parser.add_argument(
        "--rollout-probabilities",
        type=_csv_floats,
        default=(0.5, 0.25, 0.25),
    )
    parser.add_argument("--rollout-seed", type=int, default=73_001)
    parser.add_argument("--flow-time-frequency-embedding-size", type=int, default=256)
    parser.add_argument("--lambda-eulerian", type=float, default=0.0)
    parser.add_argument("--lambda-semigroup", type=float, default=0.0)
    parser.add_argument("--semigroup-middle-time", type=float, default=0.5)
    parser.add_argument("--semigroup-end-time", type=float, default=1.0)
    parser.add_argument("--eulerian-local-delta", type=float, default=0.05)
    parser.add_argument("--eulerian-end-time", type=float, default=1.0)
    parser.add_argument("--variant", default="terminal_gmd_plus_stopped_structural")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    invoked = sys.argv[1:] if argv is None else argv
    args.command = shlex.join([sys.argv[0], *invoked])
    return args


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _source_hashes() -> dict[str, str]:
    missing = [relative for relative in SOURCE_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"required structural trainer sources are absent: {missing}")
    return {relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES}


def verify_source_manifest(args: argparse.Namespace) -> dict[str, str]:
    path = args.expected_source_manifest
    digest = args.expected_source_manifest_sha256
    if path is None or digest is None:
        raise ValueError("structural training requires a source manifest and SHA-256")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("source manifest must be a regular file")
    if not _is_sha256(digest) or sha256_file(path) != digest:
        raise ValueError("source manifest external SHA-256 differs")
    artifacts = (
        args.feature_checkpoint,
        args.parent_checkpoint,
        CONVNEXT_CONFIG,
        CONVNEXT_WEIGHTS,
    )
    expected_paths = [*SOURCE_FILES, *(str(artifact) for artifact in artifacts)]
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(expected_paths):
        raise ValueError("source/artifact manifest entry count differs")
    parsed: dict[str, str] = {}
    for line, expected in zip(lines, expected_paths, strict=True):
        observed_digest, separator, observed_path = line.partition("  ")
        if (
            separator != "  "
            or not _is_sha256(observed_digest)
            or observed_path != expected
        ):
            raise ValueError(f"source manifest entry differs for {expected}")
        parsed[observed_path] = observed_digest
    sources = {relative: parsed[relative] for relative in SOURCE_FILES}
    if sources != _source_hashes():
        raise RuntimeError("structural trainer sources changed after launcher hashing")
    expected_artifacts = {
        str(args.feature_checkpoint): official.MAE_SHA256,
        str(args.parent_checkpoint): PARENT_SHA256,
        str(CONVNEXT_CONFIG): HF_CONFIG_SHA256,
        str(CONVNEXT_WEIGHTS): HF_WEIGHTS_SHA256,
    }
    if any(parsed[name] != expected for name, expected in expected_artifacts.items()):
        raise ValueError("a pinned model artifact digest differs")
    return {**sources, **{name: parsed[name] for name in expected_artifacts}}


def validate_args(args: argparse.Namespace) -> None:
    if args.smoke:
        raise ValueError("use synthetic tests for smoke coverage; production shape is pinned")
    if (
        args.data_dir.resolve() != official.TRAIN_DIR.resolve()
        or (args.train_start_index, args.train_end_index) != (0, 162_770)
    ):
        raise ValueError("structural Flow-GMD requires the official 162770-image split")
    if (
        args.feature_checkpoint.resolve() != official.MAE_PATH.resolve()
        or args.feature_checkpoint_sha256 != official.MAE_SHA256
    ):
        raise ValueError("structural Flow-GMD requires the pinned official EMA MAE")
    if (
        args.expected_feature_training_signature_sha256
        != official.MAE_SIGNATURE_SHA256
        or args.feature_weight_kind != "ema"
    ):
        raise ValueError("structural Flow-GMD requires the pinned MAE signature")
    if (
        args.parent_checkpoint.resolve() != PARENT.resolve()
        or args.parent_checkpoint_sha256 != PARENT_SHA256
    ):
        raise ValueError("structural Flow-GMD requires the validated 60k parent")
    if args.convnext_snapshot_dir.resolve() != CONVNEXT_DIR.resolve():
        raise ValueError("structural Flow-GMD requires the pinned ConvNeXt snapshot")
    for path, label in (
        (args.data_dir, "data directory"),
        (args.convnext_snapshot_dir, "ConvNeXt snapshot"),
    ):
        if not path.is_dir() or path.is_symlink():
            raise FileNotFoundError(f"{label} must be a regular directory")
    for path, label in (
        (args.feature_checkpoint, "MAE"),
        (args.parent_checkpoint, "parent"),
        (CONVNEXT_CONFIG, "ConvNeXt config"),
        (CONVNEXT_WEIGHTS, "ConvNeXt weights"),
    ):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"{label} must be a regular file")
    if len(args.rollout_nfes) != len(args.rollout_probabilities):
        raise ValueError("rollout NFEs and probabilities must have equal lengths")
    if not math.isclose(
        sum(args.rollout_probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("rollout probabilities must sum to one")
    if args.rollout_seed < 0 or args.flow_time_frequency_embedding_size < 2:
        raise ValueError("invalid rollout seed or time-embedding size")
    for value, label in (
        (args.lambda_eulerian, "Eulerian lambda"),
        (args.lambda_semigroup, "semigroup lambda"),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{label} must be finite and non-negative")
    if not (
        math.isfinite(args.semigroup_middle_time)
        and math.isfinite(args.semigroup_end_time)
        and 0.0 < args.semigroup_middle_time < args.semigroup_end_time <= 1.0
    ):
        raise ValueError("semigroup times must satisfy 0 < middle < end <= 1")
    if not (
        math.isfinite(args.eulerian_local_delta)
        and args.eulerian_local_delta > 0.0
        and math.isfinite(args.eulerian_end_time)
        and 0.0 < args.eulerian_end_time <= 1.0
    ):
        raise ValueError("Eulerian delta/end time is invalid")
    for k in args.rollout_nfes:
        source_index = min(1, k - 1)
        start = source_index / k
        if start + args.eulerian_local_delta >= args.eulerian_end_time:
            raise ValueError(
                "Eulerian schedule must satisfy source < local end < final end for every K"
            )
    if args.steps < 1 or args.log_every < 1 or args.checkpoint_every < 0:
        raise ValueError("invalid training/log/checkpoint step count")
    if (
        args.global_generated != 64
        or args.positive_count != 128
        or args.negative_count != 32
        or args.bank_push != 128
    ):
        raise ValueError("official supports must remain 64/128/32 with bank push 128")
    if tuple(args.temperatures) != (0.2, 0.05, 0.02):
        raise ValueError("official GMD temperatures must remain 0.2,0.05,0.02")
    if (
        args.augmentation != "aligned_flip"
        or not args.preload_uint8
        or not args.use_bfloat16
        or not args.deterministic
        or args.allow_short_feature_checkpoint_preflight
    ):
        raise ValueError(
            "production requires aligned flip, uint8 preload, BF16, deterministic "
            "algorithms, and full artifact preflight"
        )
    if args.device.type != "cuda" or args.distributed_backend != "nccl":
        raise ValueError("production structural Flow-GMD requires ROCm and NCCL/RCCL")
    if args.world_size != 8:
        raise ValueError("the audited structural recipe requires exactly eight ranks")
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
        raise ValueError(f"generator CLI differs from the pinned parent: {changed}")
    for count in (args.global_generated, args.positive_count, args.negative_count):
        if count % args.world_size:
            raise ValueError("all support counts must be divisible by world size")
    args.local_generated = args.global_generated // args.world_size
    if args.positive_bank_size != 128 or args.negative_bank_size != 1_000:
        raise ValueError("official positive/negative banks must remain 128/1000")
    if args.bank_push > min(args.positive_bank_size, args.negative_bank_size):
        raise ValueError("bank push exceeds a bank capacity")
    for value, label in (
        (args.lr, "lr"),
        (args.grad_clip, "grad clip"),
        (args.adam_eps, "Adam epsilon"),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{label} must be finite and positive")
    if not 0 <= args.lr_warmup_start <= args.lr or args.lr_warmup_steps < 0:
        raise ValueError("invalid learning-rate warmup")
    if not (0.0 < args.adam_beta1 < 1.0 and 0.0 < args.adam_beta2 < 1.0):
        raise ValueError("Adam betas must lie strictly between zero and one")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and non-negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("EMA decay must lie strictly between zero and one")
    if args.resume is None and args.resume_sha256 is not None:
        raise ValueError("resume SHA requires --resume")
    if args.resume is not None and (
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
        raise RuntimeError("structural Flow-GMD training requires ROCm GPUs")
    torch.cuda.set_device(args.local_rank)
    args.device = torch.device("cuda", args.local_rank)
    if args.distributed:
        dist.init_process_group(
            "nccl", timeout=timedelta(minutes=args.distributed_timeout_minutes)
        )
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
    if args.distributed:
        payload = [failure, result]
        dist.broadcast_object_list(payload, src=0)
        failure, result = payload
    if failure:
        raise RuntimeError(f"rank-zero operation failed:\n{failure}")
    barrier(args)
    return result


def gather(local: Any, args: argparse.Namespace) -> list[Any]:
    if not args.distributed:
        return [local]
    values: list[Any] = [None] * args.world_size
    dist.all_gather_object(values, local)
    if any(value is None for value in values):
        raise RuntimeError("failed to gather every rank object")
    return values


def _parent_integrity(payload: Mapping[str, Any]) -> dict[str, str]:
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
        "training_signature_sha256": canonical_json_sha256(
            dict(payload["training_signature"])
        ),
    }


def verify_parent(path: Path) -> dict[str, Any]:
    if sha256_file(path) != PARENT_SHA256:
        raise ValueError("parent checkpoint external SHA-256 differs")
    sidecar = Path(f"{path}.sha256")
    if (
        not sidecar.is_file()
        or sidecar.is_symlink()
        or sidecar.read_text(encoding="utf-8").strip()
        != f"{PARENT_SHA256}  {path.name}"
    ):
        raise ValueError("parent checkpoint sidecar differs")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {
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
        "training_signature",
        "run_dir",
        "saved_at_utc",
        "integrity",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("parent checkpoint schema differs")
    if (
        payload["format"] != "driftmodel.celeba64_direct_pixel_gmd_training"
        or payload["version"] != 2
        or payload["completed_step"] != PARENT_STEP
    ):
        raise ValueError("parent format/version/step differs")
    if dict(payload["integrity"]) != _parent_integrity(payload):
        raise ValueError("parent internal integrity differs")
    signature = payload["training_signature"]
    if (
        canonical_json_sha256(dict(signature)) != PARENT_SIGNATURE_SHA256
        or state_dict_sha256(payload["ema"]) != PARENT_EMA_SHA256
    ):
        raise ValueError("parent signature or EMA state differs")
    if (
        signature.get("objective") != PARENT_OBJECTIVE
        or signature.get("objective_implementation")
        != "src.celeba_gmd_sharded_features.global_sharded_mae_convnext_loss"
    ):
        raise ValueError("parent objective differs")
    if (
        signature.get("dataset_files_sha256") != TRAIN_FILES_SHA256
        or signature.get("dataset_image_count") != 162_770
        or signature.get("train_start_index") != 0
        or signature.get("train_end_index") != 162_770
    ):
        raise ValueError("parent dataset differs")
    support = signature.get("support", {})
    if (
        support.get("generated_global"),
        support.get("positive_count"),
        support.get("fixed_negative_count"),
    ) != (64, 128, 32):
        raise ValueError("parent support contract differs")
    return {
        "checkpoint": str(path.resolve()),
        "file_sha256": PARENT_SHA256,
        "completed_step": PARENT_STEP,
        "ema_state_sha256": PARENT_EMA_SHA256,
        "training_signature_sha256": PARENT_SIGNATURE_SHA256,
        "weights": "ema",
        "parent_optimizer_state_imported": False,
        "parent_stream_state_imported": False,
        "parent_banks_imported": False,
        "parent_rng_state_imported": False,
    }


def load_initial_flow(args: argparse.Namespace):
    payload = torch.load(
        args.parent_checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    return flow_from_pinned_parent_payload(
        payload,
        expected_step=PARENT_STEP,
        expected_objective=PARENT_OBJECTIVE,
        expected_ema_state_sha256=PARENT_EMA_SHA256,
        expected_training_signature_sha256=PARENT_SIGNATURE_SHA256,
        flow_time_frequency_embedding_size=args.flow_time_frequency_embedding_size,
    )


def choose_rollout_nfe(args: argparse.Namespace, generator: torch.Generator) -> int:
    probabilities = torch.tensor(args.rollout_probabilities, dtype=torch.float64)
    local_index = int(torch.multinomial(probabilities, 1, generator=generator).item())
    local_k = int(args.rollout_nfes[local_index])
    selected = torch.tensor(local_k, device=args.device, dtype=torch.int64)
    if args.distributed:
        dist.broadcast(selected, src=0)
    if int(selected.item()) != local_k:
        raise RuntimeError("rollout RNG diverged across ranks")
    return local_k


def structural_forward_arguments(
    k: int,
    args: argparse.Namespace,
    device: torch.device | str,
) -> tuple[Tensor, dict[str, Tensor | int]]:
    """Build the deterministic caller-owned grid and enabled branch inputs."""

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive Python integer")
    grid = torch.linspace(0.0, 1.0, k + 1, device=device, dtype=torch.float32)
    kwargs: dict[str, Tensor | int] = {}
    if args.lambda_semigroup > 0.0:
        kwargs.update(
            semigroup_middle_time=torch.tensor(
                args.semigroup_middle_time, device=device, dtype=torch.float32
            ),
            semigroup_end_time=torch.tensor(
                args.semigroup_end_time, device=device, dtype=torch.float32
            ),
        )
    if args.lambda_eulerian > 0.0:
        source_index = min(1, k - 1)
        start = grid[source_index]
        kwargs.update(
            eulerian_rollout_index=source_index,
            eulerian_start_time=start,
            eulerian_local_end_time=start
            + torch.tensor(
                args.eulerian_local_delta, device=device, dtype=torch.float32
            ),
            eulerian_end_time=torch.tensor(
                args.eulerian_end_time, device=device, dtype=torch.float32
            ),
        )
    return grid, kwargs


def weighted_objective(
    gmd_loss: Tensor,
    result: Mapping[str, Tensor | None],
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Combine enabled raw losses without ever evaluating a zero-weight branch."""

    if not isinstance(gmd_loss, Tensor) or gmd_loss.ndim != 0:
        raise ValueError("GMD loss must be a scalar Tensor")
    losses = structural_losses(result)
    expected: set[str] = set()
    if args.lambda_eulerian > 0.0:
        expected.add("eulerian_loss")
    if args.lambda_semigroup > 0.0:
        expected.add("semigroup_loss")
    if set(losses) != expected:
        raise RuntimeError("enabled structural losses differ from configured lambdas")
    zero = gmd_loss.new_zeros(())
    eulerian = losses.get("eulerian_loss", zero)
    semigroup = losses.get("semigroup_loss", zero)
    weighted_eulerian = eulerian * args.lambda_eulerian
    weighted_semigroup = semigroup * args.lambda_semigroup
    total = gmd_loss + weighted_eulerian + weighted_semigroup
    return total, {
        "total_loss": total,
        "eulerian_loss": eulerian,
        "semigroup_loss": semigroup,
        "weighted_eulerian_loss": weighted_eulerian,
        "weighted_semigroup_loss": weighted_semigroup,
    }


def resolve_run_dir(args: argparse.Namespace) -> Path:
    def resolve() -> str:
        if args.resume is not None:
            parent = args.resume.resolve().parent
            inferred = parent.parent if parent.name == "checkpoints" else parent
            if args.run_dir is not None and args.run_dir.resolve() != inferred:
                raise ValueError("run directory differs from resume checkpoint")
            if not inferred.is_dir():
                raise FileNotFoundError("resume run directory is absent")
            return str(inferred)
        if args.run_dir is not None:
            path = args.run_dir.resolve()
            if path.exists() and any(path.iterdir()):
                raise FileExistsError("new run directory is not empty")
            path.mkdir(parents=True, exist_ok=True)
            return str(path)
        label = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in args.run_label
        ).strip("_")
        path = (
            args.output_root
            / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{label}"
        ).resolve()
        path.mkdir(parents=True, exist_ok=False)
        return str(path)

    return Path(rank_zero(args, resolve))


def training_signature(
    args,
    dataset_manifest,
    flow,
    feature_identity,
    feature_encoder,
    parent_identity,
) -> dict[str, Any]:
    return {
        "version": SIGNATURE_VERSION,
        "experiment": "celeba_flow_gmd_structural_20260914",
        "variant": args.variant,
        "objective": (
            "official_terminal_GMD_MAE82_ConvNeXt13_plus_stopped_teacher_"
            "Eulerian_and_semigroup"
        ),
        "objective_implementation": {
            "terminal": (
                "src.celeba_gmd_sharded_features."
                "global_sharded_mae_convnext_loss"
            ),
            "structural": "src.celeba_gmd_structural_rollout.StructuralComposedRollout",
        },
        "flow": flow.flow_architecture_config(),
        "initialization": parent_identity,
        "dataset": dict(dataset_manifest),
        "feature_checkpoint": dict(feature_identity),
        "feature_representation": composite_representation_signature(feature_encoder),
        "supports": {
            "generated_global": 64,
            "generated_local": args.local_generated,
            "positive": 128,
            "fixed_negative": 32,
            "positive_bank": args.positive_bank_size,
            "negative_bank": args.negative_bank_size,
            "bank_push": 128,
        },
        "rollout": {
            "nfe_values": list(args.rollout_nfes),
            "probabilities": list(args.rollout_probabilities),
            "selection": (
                "one identical CPU categorical draw per rank, rank0 broadcast "
                "and equality assertion"
            ),
            "seed": args.rollout_seed,
            "grid": "caller-supplied uniform float32 [K+1]",
            "fixed_noise_labels": True,
            "noise_reinjection": False,
            "terminal_clip": False,
        },
        "randomness": {
            "base_seed": args.seed,
            "parameter_seed": args.seed,
            "global_bank_stream_seed": args.seed + 101,
            "bank_and_aligned_flip_seed": args.seed + 29,
            "rank_prior_seed_rule": "base_seed + 1000003 * rank + 17",
            "rollout_seed": args.rollout_seed,
            "deterministic_algorithms": True,
        },
        "structural": {
            "lambda_eulerian": args.lambda_eulerian,
            "lambda_semigroup": args.lambda_semigroup,
            "disabled_when_lambda_zero": True,
            "teachers": "no_grad_and_detached",
            "semigroup_times": [
                args.semigroup_middle_time,
                args.semigroup_end_time,
            ],
            "eulerian_source_index": "min(1,K-1)",
            "eulerian_local_delta": args.eulerian_local_delta,
            "eulerian_end_time": args.eulerian_end_time,
            "shared_rank_local_prior_and_noise_labels": True,
        },
        "temperatures": list(args.temperatures),
        "optimizer": {
            "name": "AdamW",
            "fresh_at_flow_step_zero": True,
            "lr": args.lr,
            "warmup_start": args.lr_warmup_start,
            "warmup_steps": args.lr_warmup_steps,
            "betas": [args.adam_beta1, args.adam_beta2],
            "eps": args.adam_eps,
            "weight_decay": args.weight_decay,
            "grad_clip": args.grad_clip,
        },
        "ema": {"decay": args.ema_decay, "initialized_from": "flow_theta_0"},
        "precision": (
            "FP32 parameters/optimizer/loss accumulation; BF16 flow+MAE "
            "autocast; ConvNeXt FP32"
        ),
        "distributed": {
            "world_size": args.world_size,
            "wrapper": "one outer StructuralComposedRollout DDP forward per update",
            "static_graph": False,
            "broadcast_buffers": False,
            "find_unused_parameters": False,
            "feature_sharding": True,
        },
        "source_and_artifact_sha256": dict(args.launcher_source_hashes),
        "claim_limit": (
            "terminal distribution plus stopped-teacher structural regularization; "
            "not an exact semigroup or Eulerian identity claim"
        ),
    }


def _snapshot(
    run_dir: Path, expected: Mapping[str, str], name: str
) -> dict[str, Any]:
    destination = run_dir / name
    if destination.exists():
        raise FileExistsError(destination)
    before = _source_hashes()
    expected_sources = {relative: expected[relative] for relative in SOURCE_FILES}
    if before != expected_sources:
        raise RuntimeError("source snapshot differs from launcher manifest")
    for relative in SOURCE_FILES:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    copied = {
        relative: sha256_file(destination / relative) for relative in SOURCE_FILES
    }
    if copied != before or _source_hashes() != before:
        raise RuntimeError("sources changed while snapshotting")
    manifest = {
        "format": "driftmodel.celeba64_flow_gmd_structural_source_snapshot",
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": copied,
    }
    path = atomic_json_dump(destination / "manifest.json", manifest)
    return {
        **manifest,
        "manifest": str(path.resolve()),
        "manifest_sha256": sha256_file(path),
    }


def write_metadata(args, run_dir, dataset_manifest, signature, flow) -> None:
    suffix = (
        ""
        if args.resume is None
        else f"_resume_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    snapshot = _snapshot(
        run_dir, args.launcher_source_hashes, f"source_snapshot{suffix}"
    )
    atomic_json_dump(run_dir / f"dataset_manifest{suffix}.json", dataset_manifest)
    config = {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()
        if not key.startswith("_")
    }
    config.update(
        parameter_count=sum(parameter.numel() for parameter in flow.parameters()),
        training_signature=signature,
    )
    atomic_json_dump(
        run_dir / f"config{suffix}.json",
        json.loads(json.dumps(config, default=str)),
    )
    atomic_json_dump(
        run_dir / f"provenance{suffix}.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "command": args.command,
            "hostname": socket.gethostname(),
            "torch": str(torch.__version__),
            "hip": getattr(torch.version, "hip", None),
            "world_size": args.world_size,
            "source_snapshot": snapshot,
            "pinned_artifacts": {
                name: digest
                for name, digest in args.launcher_source_hashes.items()
                if name not in SOURCE_FILES
            },
            "parent": signature["initialization"],
        },
    )


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
        "rollout_rng_state_sha256": torch_tree_sha256(payload["rollout_rng_state"]),
        "rollout_counts_sha256": torch_tree_sha256(payload["rollout_counts"]),
        "training_signature_sha256": canonical_json_sha256(
            dict(payload["training_signature"])
        ),
    }


def _assert_replicated_training_state(model, ema, optimizer, args) -> None:
    local = {
        "model": state_dict_sha256(
            unwrap_structural_composed_rollout(model).flow.state_dict()
        ),
        "ema": state_dict_sha256(ema.state_dict()),
        "optimizer": torch_tree_sha256(optimizer.state_dict()),
    }
    replicas = gather(local, args)
    if any(replica != replicas[0] for replica in replicas[1:]):
        raise RuntimeError("model/EMA/optimizer state differs across DDP ranks")


def save_checkpoint(
    path,
    *,
    completed_step,
    model,
    ema,
    optimizer,
    stream,
    positive_bank,
    negative_bank,
    bank_generator,
    prior_generator,
    rollout_generator,
    rollout_counts,
    signature,
    run_dir,
    args,
) -> str:
    _assert_replicated_training_state(model, ema, optimizer, args)
    prior_states = gather(prior_generator.get_state().cpu(), args)
    globals_by_rank = gather(
        {
            "stream": stream.state_dict(),
            "positive": positive_bank.state_dict(),
            "negative": negative_bank.state_dict(),
            "bank_rng": bank_generator.get_state().cpu(),
            "rollout_rng": rollout_generator.get_state().cpu(),
            "rollout_counts": dict(rollout_counts),
        },
        args,
    )
    shared_hashes = [torch_tree_sha256(value) for value in globals_by_rank]
    if len(set(shared_hashes)) != 1:
        raise RuntimeError("global stream/banks/K RNG diverged across ranks")
    shared = globals_by_rank[0]

    def save() -> str:
        flow = unwrap_structural_composed_rollout(model).flow
        payload = {
            "format": TRAINING_FORMAT,
            "version": TRAINING_VERSION,
            "completed_step": completed_step,
            "model": {
                name: value.detach().cpu() for name, value in flow.state_dict().items()
            },
            "ema": {
                name: value.detach().cpu() for name, value in ema.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "data_stream": shared["stream"],
            "positive_bank": shared["positive"],
            "negative_bank": shared["negative"],
            "bank_rng_state": shared["bank_rng"],
            "prior_rng_by_rank": prior_states,
            "rollout_rng_state": shared["rollout_rng"],
            "rollout_counts": shared["rollout_counts"],
            "training_signature": dict(signature),
            "run_dir": str(run_dir.resolve()),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        payload["integrity"] = _checkpoint_integrity(payload)
        atomic_torch_save(path, payload)
        digest = sha256_file(path)
        atomic_text_write(Path(f"{path}.sha256"), f"{digest}  {path.name}\n")
        return digest

    return str(rank_zero(args, save))


def load_checkpoint(
    path,
    expected_sha,
    *,
    model,
    ema,
    optimizer,
    stream,
    positive_bank,
    negative_bank,
    bank_generator,
    prior_generator,
    rollout_generator,
    rollout_counts,
    signature,
    run_dir,
    args,
) -> int:
    if sha256_file(path) != expected_sha:
        raise ValueError("resume external SHA-256 differs")
    sidecar = Path(f"{path}.sha256")
    if (
        not sidecar.is_file()
        or sidecar.is_symlink()
        or sidecar.read_text(encoding="utf-8").strip()
        != f"{expected_sha}  {path.name}"
    ):
        raise ValueError("resume sidecar differs")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = {
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
        "rollout_rng_state",
        "rollout_counts",
        "training_signature",
        "run_dir",
        "saved_at_utc",
        "integrity",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected
        or payload["format"] != TRAINING_FORMAT
        or payload["version"] != TRAINING_VERSION
    ):
        raise ValueError("resume structural Flow-GMD schema differs")
    if dict(payload["integrity"]) != _checkpoint_integrity(payload):
        raise ValueError("resume internal integrity differs")
    if (
        payload["training_signature"] != dict(signature)
        or Path(payload["run_dir"]).resolve() != run_dir.resolve()
    ):
        raise ValueError("resume signature or run directory differs")
    if (
        type(payload["completed_step"]) is not int
        or payload["completed_step"] < 1
        or not isinstance(payload["prior_rng_by_rank"], list)
        or len(payload["prior_rng_by_rank"]) != args.world_size
    ):
        raise ValueError("resume step or rank-local RNG schema differs")
    expected_count_keys = {str(k) for k in args.rollout_nfes}
    if (
        not isinstance(payload["rollout_counts"], Mapping)
        or set(payload["rollout_counts"]) != expected_count_keys
        or any(
            type(value) is not int or value < 0
            for value in payload["rollout_counts"].values()
        )
        or sum(payload["rollout_counts"].values()) != payload["completed_step"]
    ):
        raise ValueError("resume rollout counts differ from the completed step")
    unwrap_structural_composed_rollout(model).flow.load_state_dict(
        payload["model"], strict=True
    )
    ema.load_state_dict(payload["ema"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    stream.load_state_dict(payload["data_stream"])
    positive_bank.load_state_dict(payload["positive_bank"])
    negative_bank.load_state_dict(payload["negative_bank"])
    bank_generator.set_state(payload["bank_rng_state"].cpu())
    prior_generator.set_state(payload["prior_rng_by_rank"][args.rank].cpu())
    rollout_generator.set_state(payload["rollout_rng_state"].cpu())
    rollout_counts.clear()
    rollout_counts.update(
        {str(k): int(value) for k, value in payload["rollout_counts"].items()}
    )
    _assert_replicated_training_state(model, ema, optimizer, args)
    return int(payload["completed_step"])


def verify_gradients(
    model: nn.Module,
    feature_encoder: FrozenMAEConvNextFeatures,
    args,
) -> None:
    gradients = [parameter.grad for parameter in model.parameters()]
    feature_ok = not any(
        parameter.grad is not None for parameter in feature_encoder.parameters()
    )
    complete = all(gradient is not None for gradient in gradients)
    finite = complete and all(
        bool(torch.isfinite(gradient).all())
        for gradient in gradients
        if gradient is not None
    )
    pixels = feature_encoder.last_local_generated
    pixel_ok = pixels is not None and pixels.grad is not None
    if pixel_ok:
        assert pixels is not None and pixels.grad is not None
        pixel_ok = bool(torch.isfinite(pixels.grad).all()) and bool(
            pixels.grad.abs().sum() > 0
        )
    flags = torch.tensor(
        [feature_ok, complete, finite, pixel_ok],
        device=args.device,
        dtype=torch.int32,
    )
    if args.distributed:
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
    feature_encoder.clear_step_state()
    labels = (
        "frozen feature parameters",
        "complete flow gradients",
        "finite flow gradients",
        "nonzero terminal RGB gradient",
    )
    failures = [label for label, passed in zip(labels, flags.tolist()) if not passed]
    if failures:
        raise FloatingPointError(
            f"detailed gradient verification failed: {', '.join(failures)}"
        )


def export_artifacts(
    run_dir,
    completed_step,
    model,
    ema,
    signature,
    checkpoint,
    checkpoint_sha,
) -> None:
    def metadata(kind):
        return {
            "completed_step": completed_step,
            "weight_kind": kind,
            "training_signature": dict(signature),
            "training_signature_sha256": canonical_json_sha256(dict(signature)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    raw_path = run_dir / "flow_raw_final.pt"
    ema_path = run_dir / "flow_ema_final.pt"
    raw_rollout = ComposedRollout(unwrap_structural_composed_rollout(model).flow)
    atomic_torch_save(
        raw_path,
        composed_rollout_artifact_dict(raw_rollout, metadata=metadata("raw")),
    )
    atomic_torch_save(
        ema_path,
        composed_rollout_artifact_dict(ComposedRollout(ema), metadata=metadata("ema")),
    )
    load_composed_rollout_artifact(raw_path, "cpu")
    load_composed_rollout_artifact(ema_path, "cpu")
    raw_sha = sha256_file(raw_path)
    ema_sha = sha256_file(ema_path)
    atomic_text_write(Path(f"{raw_path}.sha256"), f"{raw_sha}  {raw_path.name}\n")
    atomic_text_write(Path(f"{ema_path}.sha256"), f"{ema_sha}  {ema_path.name}\n")
    manifest = {
        "format": "driftmodel.celeba64_flow_gmd_structural_artifacts",
        "version": 1,
        "completed_step": completed_step,
        "strict_reload": True,
        "artifacts": {
            "training_checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_sha,
            },
            "raw": {"path": str(raw_path.resolve()), "sha256": raw_sha},
            "ema": {"path": str(ema_path.resolve()), "sha256": ema_sha},
        },
    }
    path = atomic_json_dump(run_dir / "artifact_manifest.json", manifest)
    atomic_text_write(
        run_dir / "artifact_manifest.json.sha256",
        f"{sha256_file(path)}  {path.name}\n",
    )


def train(args: argparse.Namespace) -> Path:
    parent_identity = rank_zero(args, lambda: verify_parent(args.parent_checkpoint))
    dataset_manifest = rank_zero(
        args,
        lambda: build_dataset_manifest(args.data_dir.resolve(), 0, 162_770),
    )
    if (
        dataset_manifest["files_sha256"] != TRAIN_FILES_SHA256
        or dataset_manifest["image_count"] != 162_770
    ):
        raise ValueError("official dataset manifest differs")
    dataset = CelebA64PixelDataset(
        args.data_dir.resolve(), dataset_manifest, preload_uint8=True
    )
    if (
        rank_zero(
            args,
            lambda: build_dataset_manifest(args.data_dir.resolve(), 0, 162_770),
        )
        != dataset_manifest
    ):
        raise RuntimeError("dataset changed while preloading")
    stream = base.GlobalBankStream(dataset, batch_size=128, seed=args.seed + 101)
    positive_bank = PositiveImageBank(args.positive_bank_size)
    negative_bank = NegativeImageBank(args.negative_bank_size)
    bank_generator = base.make_bank_generator(args)
    feature_identity = base.feature_checkpoint_identity(args)
    mae = load_frozen_pixel_feature_encoder(args.feature_checkpoint, args.device)
    convnext = load_local_convnextv2_base(
        args.convnext_snapshot_dir, device=args.device
    )
    feature_encoder = FrozenMAEConvNextFeatures(mae, convnext).eval()

    base.seed_parameters(args)
    flow = load_initial_flow(args).to(args.device)
    ema = copy.deepcopy(flow).eval().requires_grad_(False)
    rollout = StructuralComposedRollout(flow).train()
    train_model: nn.Module = rollout
    if args.distributed:
        train_model = DistributedDataParallel(
            rollout,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            static_graph=False,
        )
    prior_generator = base.make_prior_generator(args)
    rollout_generator = torch.Generator(device="cpu").manual_seed(args.rollout_seed)
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    if optimizer.state:
        raise RuntimeError("structural Flow-GMD optimizer must start empty")
    signature = training_signature(
        args,
        dataset_manifest,
        flow,
        feature_identity,
        feature_encoder,
        parent_identity,
    )
    run_dir = resolve_run_dir(args)
    rollout_counts = {str(value): 0 for value in args.rollout_nfes}
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(
            args.resume,
            args.resume_sha256,
            model=train_model,
            ema=ema,
            optimizer=optimizer,
            stream=stream,
            positive_bank=positive_bank,
            negative_bank=negative_bank,
            bank_generator=bank_generator,
            prior_generator=prior_generator,
            rollout_generator=rollout_generator,
            rollout_counts=rollout_counts,
            signature=signature,
            run_dir=run_dir,
            args=args,
        )
    if start_step >= args.steps:
        raise ValueError("--steps must exceed the resumed structural flow step")
    rank_zero(
        args,
        lambda: write_metadata(args, run_dir, dataset_manifest, signature, flow),
    )
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    if args.rank == 0:
        logger.log(
            {
                "event": "resume" if start_step else "start",
                "step": start_step,
                "world_size": args.world_size,
                "rollout_nfes": list(args.rollout_nfes),
                "rollout_probabilities": list(args.rollout_probabilities),
                "lambda_eulerian": args.lambda_eulerian,
                "lambda_semigroup": args.lambda_semigroup,
                "parent_step": PARENT_STEP,
                "fresh_optimizer": start_step == 0,
                "parameter_count": sum(parameter.numel() for parameter in flow.parameters()),
            }
        )

    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.monotonic()
    progress = trange(
        start_step,
        args.steps,
        initial=start_step,
        total=args.steps,
        desc="celeba64:flow-gmd-structural",
        disable=args.rank != 0,
    )
    final_path = run_dir / "checkpoint_final.pt"
    for step_index in progress:
        update_started = time.monotonic()
        lr = base.learning_rate_at_step(step_index, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        pushed = base.augment_bank_images(
            stream.next(), "aligned_flip", generator=bank_generator
        )
        positive_bank.add(pushed)
        negative_bank.add(pushed)
        positives = positive_bank.sample(128, generator=bank_generator).to(
            args.device, non_blocking=True
        )
        negatives = negative_bank.sample(32, generator=bank_generator).to(
            args.device, non_blocking=True
        )
        structural = unwrap_structural_composed_rollout(train_model)
        prior, labels = structural.flow.sample_prior(
            args.local_generated,
            device=args.device,
            generator=prior_generator,
        )
        k = choose_rollout_nfe(args, rollout_generator)
        rollout_counts[str(k)] += 1
        rollout_grid, structural_kwargs = structural_forward_arguments(
            k, args, args.device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = train_model(
                prior,
                labels,
                k,
                rollout_grid,
                **structural_kwargs,
            )
            terminal = result["terminal"]
            if not isinstance(terminal, Tensor):
                raise RuntimeError("structural wrapper omitted the terminal Tensor")
            gmd_loss, gmd_details = global_sharded_mae_convnext_loss(
                feature_encoder,
                terminal,
                positives,
                negatives,
                temperatures=args.temperatures,
            )
            total_loss, objective_metrics = weighted_objective(
                gmd_loss, result, args
            )
        finite_values = torch.stack(
            [value.detach().float() for value in objective_metrics.values()]
        )
        finite = torch.isfinite(finite_values).all().to(torch.int32)
        if args.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(
                f"non-finite structural objective at step {step_index + 1}"
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
            verify_gradients(train_model, feature_encoder, args)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            train_model.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        base.update_ema(
            ema,
            unwrap_structural_composed_rollout(train_model).flow,
            args.ema_decay,
        )
        if completed == 1 or completed == args.steps or completed % args.log_every == 0:
            torch.cuda.synchronize(args.device)
            zero = total_loss.detach().new_zeros(())
            reduced = base.reduce_stats(
                {
                    **base.compact_gmd_statistics(gmd_details),
                    **objective_metrics,
                    "eulerian_rmse": result["eulerian_rmse"]
                    if isinstance(result["eulerian_rmse"], Tensor)
                    else zero,
                    "semigroup_rmse": result["semigroup_rmse"]
                    if isinstance(result["semigroup_rmse"], Tensor)
                    else zero,
                    "gradient_norm_total_preclip": gradient_norm,
                    "terminal_mean": terminal.detach().float().mean(),
                    "terminal_std": terminal.detach().float().std(correction=0),
                },
                args,
            )
            maxima = base.reduce_max_stats(
                {
                    "seconds_per_update_max_rank": time.monotonic()
                    - update_started,
                    "gpu_peak_memory_allocated_bytes_max_rank": (
                        torch.cuda.max_memory_allocated(args.device)
                    ),
                    "gpu_peak_memory_reserved_bytes_max_rank": (
                        torch.cuda.max_memory_reserved(args.device)
                    ),
                },
                args,
            )
            source_index = min(1, k - 1) if args.lambda_eulerian > 0.0 else None
            record = {
                "event": "train",
                "step": completed,
                "rollout_nfe": k,
                "rollout_counts": dict(rollout_counts),
                "eulerian_source_index": source_index,
                "eulerian_start_time": (
                    source_index / k if source_index is not None else None
                ),
                "learning_rate": lr,
                "lambda_eulerian": args.lambda_eulerian,
                "lambda_semigroup": args.lambda_semigroup,
                "elapsed_seconds_this_invocation": time.monotonic() - started,
                "detailed_gradient_check": detailed,
                **{name: float(value.cpu()) for name, value in reduced.items()},
                **{name: float(value.cpu()) for name, value in maxima.items()},
            }
            if args.rank == 0:
                logger.log(record)
                progress.set_postfix(loss=f"{record['total_loss']:.4f}", k=k)
        # Break every reference to this update's autograd graph before the
        # next RHS starts building a new structural rollout.  Checkpointing
        # below needs only persistent model/optimizer/stream/RNG state.
        feature_encoder.clear_step_state()
        del (
            result,
            terminal,
            total_loss,
            objective_metrics,
            gmd_loss,
            gmd_details,
            positives,
            negatives,
            prior,
            labels,
            rollout_grid,
            structural_kwargs,
            finite_values,
        )
        if args.checkpoint_every and completed % args.checkpoint_every == 0:
            path = run_dir / "checkpoints" / f"checkpoint_step_{completed:08d}.pt"
            digest = save_checkpoint(
                path,
                completed_step=completed,
                model=train_model,
                ema=ema,
                optimizer=optimizer,
                stream=stream,
                positive_bank=positive_bank,
                negative_bank=negative_bank,
                bank_generator=bank_generator,
                prior_generator=prior_generator,
                rollout_generator=rollout_generator,
                rollout_counts=rollout_counts,
                signature=signature,
                run_dir=run_dir,
                args=args,
            )
            if args.rank == 0:
                logger.log(
                    {
                        "event": "checkpoint",
                        "step": completed,
                        "path": str(path.resolve()),
                        "sha256": digest,
                    }
                )

    final_digest = save_checkpoint(
        final_path,
        completed_step=args.steps,
        model=train_model,
        ema=ema,
        optimizer=optimizer,
        stream=stream,
        positive_bank=positive_bank,
        negative_bank=negative_bank,
        bank_generator=bank_generator,
        prior_generator=prior_generator,
        rollout_generator=rollout_generator,
        rollout_counts=rollout_counts,
        signature=signature,
        run_dir=run_dir,
        args=args,
    )
    rank_zero(
        args,
        lambda: export_artifacts(
            run_dir,
            args.steps,
            train_model,
            ema,
            signature,
            final_path,
            final_digest,
        ),
    )
    if args.rank == 0:
        logger.log(
            {
                "event": "complete",
                "step": args.steps,
                "elapsed_seconds_this_invocation": time.monotonic() - started,
                "final_checkpoint_sha256": final_digest,
                "rollout_counts": rollout_counts,
            }
        )
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    try:
        setup_distributed(args)
        return train(args)
    finally:
        cleanup(args)


if __name__ == "__main__":
    main()
