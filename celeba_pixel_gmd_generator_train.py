#!/usr/bin/env python3
"""Auditable DDP training of a direct RGB CelebA-64 GMD generator.

This is a PyTorch, unconditional, 64px adaptation of the released ImageNet
pixel generator recipe.  It consumes a strictly verified and frozen phase-2
MAE encoder and calls ``src.official_drifting`` directly for each frozen
feature family.  There is no VAE, time integration, historical KDE objective,
or FID computation in this training program.
"""

from __future__ import annotations

import os
from pathlib import Path


def _configure_rank_local_miopen_cache() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cache_base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
    if cache_base is None and world_size > 1:
        token = os.environ.get("SLURM_JOB_ID", os.environ.get("MASTER_PORT", "local"))
        cache_base = f"/tmp/driftmodel_pixel_gmd_generator_miopen_{token}"
    if cache_base is None:
        return
    rank_root = Path(cache_base) / f"rank_{os.environ.get('LOCAL_RANK', '0')}"
    for directory in (rank_root, rank_root / "userdb", rank_root / "kernels"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MIOPEN_CACHE_DIR", str(rank_root))
    os.environ.setdefault("MIOPEN_USER_DB_PATH", str(rank_root / "userdb"))
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(rank_root / "kernels"))


_configure_rank_local_miopen_cache()

import argparse
import copy
import json
import math
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import __version__ as PIL_VERSION
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from tqdm import trange

# Reuse only the already-audited, frozen phase-2 data and serialization
# primitives.  That dependency is hashed and copied into every source snapshot.
from celeba_pixel_gmd_mae_train import (
    CelebA64PixelDataset,
    JsonlLogger,
    atomic_json_dump,
    atomic_text_write,
    atomic_torch_save,
    build_dataset_manifest,
    torch_tree_sha256,
)
from src.celeba_pixel_gmd_features import (
    canonical_json_sha256,
    load_frozen_pixel_feature_encoder,
    sha256_file,
    state_dict_sha256,
)
from src.celeba_pixel_gmd_generator import (
    DEFAULT_CONDITION_TOKENS,
    DEFAULT_DEPTH,
    DEFAULT_HEADS,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MLP_RATIO,
    DEFAULT_NOISE_CLASSES,
    DEFAULT_NOISE_COORDS,
    DEFAULT_PATCH_SIZE,
    OFFICIAL_PIXEL_GENERATOR_CONFIG,
    OFFICIAL_RELEASE_COMMIT,
    CelebA64DirectPixelGMDDiT,
    NegativeImageBank,
    PositiveImageBank,
    generator_artifact_dict,
    global_pixel_gmd_loss,
    load_pixel_gmd_generator_artifact,
)


REPO_ROOT = Path(__file__).resolve().parent
TRAINING_CHECKPOINT_FORMAT = "driftmodel.celeba64_direct_pixel_gmd_training"
TRAINING_CHECKPOINT_VERSION = 2
TRAINING_SIGNATURE_VERSION = 2
SOURCE_SNAPSHOT_FORMAT = "driftmodel.celeba64_direct_pixel_gmd_source_snapshot"
SOURCE_SNAPSHOT_VERSION = 1
ARTIFACT_MANIFEST_FORMAT = "driftmodel.celeba64_direct_pixel_gmd_artifacts"
ARTIFACT_MANIFEST_VERSION = 1
SOURCE_FILES = (
    'celeba_pixel_gmd_generator_train.py',
    'src/celeba_pixel_gmd_generator.py',
    'src/celeba_pixel_gmd_features.py',
    'src/official_drifting.py',
    'celeba_pixel_gmd_mae_train.py',
)
AUGMENTATIONS = ("aligned_flip", "none")
DEFAULT_TEMPERATURES = (0.2, 0.05, 0.02)
PRODUCTION_FEATURE_BASE_CHANNELS = 64
PRODUCTION_FEATURE_LAYERS = [3, 4, 6, 3]
PRODUCTION_FEATURE_PROFILE = "nondegenerate"
PRODUCTION_FEATURE_MINIMUM_UPDATES = 10_000
PRODUCTION_FEATURE_ARCHITECTURE = {
    "base_channels": PRODUCTION_FEATURE_BASE_CHANNELS,
    "layers": PRODUCTION_FEATURE_LAYERS,
    "dropout_prob": 0.0,
    "gn_max_groups": 32,
    "input_patch_size": 2,
    "mask_patch_size": 2,
    "feature_patch_sizes": [2, 4],
    "every_k_block": 2,
    "feature_profile": PRODUCTION_FEATURE_PROFILE,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if isinstance(value, argparse.Namespace):
        return _json_safe(vars(value))
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON provenance cannot contain non-finite floats")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported provenance value: {type(value).__name__}")


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_temperatures(raw: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("temperatures must be comma-separated floats") from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError("temperatures must be finite and positive")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Direct RGB CelebA64 generator trained with released GMD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "celeba64" / "celeba64_train",
    )
    parser.add_argument("--train-start-index", type=int, default=5_000)
    parser.add_argument("--train-end-index", type=int, default=30_000)
    parser.add_argument("--feature-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-checkpoint-sha256", required=True)
    parser.add_argument("--feature-weight-kind", choices=("ema", "raw"), default="ema")
    parser.add_argument(
        "--expected-feature-training-signature-sha256",
        default=None,
        help=(
            "required outside --smoke; pins the exact audited phase-2 training "
            "recipe independently of the artifact hash"
        ),
    )
    parser.add_argument(
        "--allow-short-feature-checkpoint-preflight",
        action="store_true",
        help=(
            "GPU preflight only: relax the 10k-update feature floor while retaining "
            "the exact production architecture and EMA requirement"
        ),
    )
    parser.add_argument(
        "--expected-source-manifest",
        type=Path,
        default=None,
        help="launcher-created pre-import SHA-256 manifest for all imported training sources",
    )
    parser.add_argument(
        "--expected-source-manifest-sha256",
        default=None,
        help="external SHA-256 pin for --expected-source-manifest",
    )

    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument(
        "--global-generated",
        type=int,
        default=64,
        help="complete generated support across all ranks",
    )
    parser.add_argument("--positive-count", type=int, default=128)
    parser.add_argument("--positive-bank-size", type=int, default=128)
    parser.add_argument("--negative-count", type=int, default=32)
    parser.add_argument("--negative-bank-size", type=int, default=1_000)
    parser.add_argument("--bank-push", type=int, default=128)
    parser.add_argument("--augmentation", choices=AUGMENTATIONS, default="aligned_flip")
    parser.add_argument(
        "--temperatures", type=_parse_temperatures, default=DEFAULT_TEMPERATURES
    )

    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--mlp-ratio", type=float, default=DEFAULT_MLP_RATIO)
    parser.add_argument("--noise-classes", type=int, default=DEFAULT_NOISE_CLASSES)
    parser.add_argument("--noise-coords", type=int, default=DEFAULT_NOISE_COORDS)
    parser.add_argument("--condition-tokens", type=int, default=DEFAULT_CONDITION_TOKENS)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr-warmup-start", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=10_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--use-bfloat16", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--preload-uint8", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--distributed-timeout-minutes", type=int, default=30)

    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "celeba_pixel_gmd" / "generator",
    )
    parser.add_argument("--run-label", default="celeba64_direct_pixel_gmd")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-sha256", default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument(
        "--verify-gradients",
        action="store_true",
        help=(
            "run the detailed complete/finite/DDP gradient fingerprint at global "
            "step 1 and then at --gradient-verification-interval"
        ),
    )
    parser.add_argument(
        "--gradient-verification-interval",
        type=int,
        default=0,
        help="0 disables periodic detailed checks after global step 1",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU/Gloo configuration; still executes the real frozen-MAE GMD path",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    invoked = sys.argv[1:] if argv is None else argv
    args.command = shlex.join([sys.argv[0], *invoked])
    if args.smoke:
        args.device = "cpu"
        args.distributed_backend = "gloo"
        args.steps = min(args.steps, 2)
        args.train_start_index = 0
        args.train_end_index = min(args.train_end_index or 16, 16)
        args.global_generated = min(args.global_generated, int(os.environ.get("WORLD_SIZE", "1")))
        args.positive_count = min(args.positive_count, 2)
        args.positive_bank_size = min(args.positive_bank_size, 4)
        args.negative_count = min(args.negative_count, 2)
        args.negative_bank_size = min(args.negative_bank_size, 4)
        args.bank_push = min(args.bank_push, 2)
        args.patch_size = 16
        args.hidden_size = 32
        args.depth = 1
        args.num_heads = 4
        args.mlp_ratio = 2.0
        args.noise_classes = 4
        args.noise_coords = 2
        args.condition_tokens = 2
        args.gradient_checkpointing = False
        args.lr_warmup_steps = 0
        args.use_bfloat16 = False
        args.preload_uint8 = False
        args.augmentation = "none"
        args.log_every = 1
        args.checkpoint_every = 1
        args.verify_gradients = True
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.is_dir() or args.data_dir.is_symlink():
        raise FileNotFoundError("--data-dir must be a regular directory, not a symlink")
    if not args.feature_checkpoint.is_file() or args.feature_checkpoint.is_symlink():
        raise FileNotFoundError("--feature-checkpoint must be a regular file")
    if not _valid_sha256(args.feature_checkpoint_sha256):
        raise ValueError("--feature-checkpoint-sha256 must be lowercase SHA-256")
    if not args.smoke and not _valid_sha256(
        args.expected_feature_training_signature_sha256
    ):
        raise ValueError(
            "non-smoke training requires --expected-feature-training-signature-sha256"
        )
    if args.smoke and args.expected_feature_training_signature_sha256 is not None and not _valid_sha256(
        args.expected_feature_training_signature_sha256
    ):
        raise ValueError("expected feature training signature must be lowercase SHA-256")
    if sha256_file(args.feature_checkpoint) != args.feature_checkpoint_sha256:
        raise ValueError("feature checkpoint external SHA-256 mismatch")
    if (args.expected_source_manifest is None) != (
        args.expected_source_manifest_sha256 is None
    ):
        raise ValueError("expected source manifest and its SHA-256 must be provided together")
    if not args.smoke and args.expected_source_manifest is None:
        raise ValueError("non-smoke training requires the launcher source manifest")
    args.launcher_expected_source_hashes = verify_launcher_source_manifest(args)
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.gradient_verification_interval < 0:
        raise ValueError("gradient verification interval must be non-negative")
    if args.feature_weight_kind != "ema":
        raise ValueError("the production pixel-GMD contract requires EMA features")
    if args.use_bfloat16 and args.device.type != "cuda":
        raise ValueError("BF16 training is supported only through CUDA/ROCm autocast")
    if args.allow_short_feature_checkpoint_preflight and (
        args.smoke or args.device.type != "cuda" or args.steps > 2
    ):
        raise ValueError(
            "short feature checkpoints are allowed only for an explicit <=2-step GPU preflight"
        )
    for name in (
        "global_generated",
        "positive_count",
        "positive_bank_size",
        "negative_count",
        "negative_bank_size",
        "bank_push",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.global_generated % args.world_size:
        raise ValueError("global generated support must be divisible by world size")
    args.local_generated = args.global_generated // args.world_size
    if args.bank_push > min(args.positive_bank_size, args.negative_bank_size):
        raise ValueError("bank push cannot exceed either bank capacity")
    if args.train_start_index < 0 or args.train_end_index <= args.train_start_index:
        raise ValueError("invalid half-open training slice")
    if not args.smoke and not args.preload_uint8:
        raise ValueError("audited non-smoke training requires --preload-uint8")
    positive_float_names = ("mlp_ratio", "lr", "adam_eps", "grad_clip")
    for name in positive_float_names:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0 <= args.lr_warmup_start <= args.lr or args.lr_warmup_steps < 0:
        raise ValueError("invalid learning-rate warmup")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight decay must be finite and non-negative")
    if not 0 < args.adam_beta1 < 1 or not 0 < args.adam_beta2 < 1:
        raise ValueError("Adam betas must lie strictly between zero and one")
    if not 0 < args.ema_decay < 1:
        raise ValueError("EMA decay must lie strictly between zero and one")
    if args.log_every < 1 or args.checkpoint_every < 0:
        raise ValueError("invalid logging/checkpoint interval")
    if args.distributed_timeout_minutes < 1:
        raise ValueError("distributed timeout must be positive")
    if args.resume is None:
        if args.resume_sha256 is not None:
            raise ValueError("--resume-sha256 is only valid with --resume")
    elif (
        not args.resume.is_file()
        or args.resume.is_symlink()
        or not _valid_sha256(args.resume_sha256)
    ):
        raise ValueError("--resume requires a regular file and lowercase SHA-256")


def setup_distributed(args: argparse.Namespace) -> None:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1
    if args.smoke:
        args.global_generated = args.world_size
        torch.set_num_threads(1)
    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested == "auto":
        requested = "cpu"
    if args.distributed:
        if requested == "cuda":
            if args.distributed_backend != "nccl" or not torch.cuda.is_available():
                raise RuntimeError("GPU DDP requires available CUDA/ROCm and NCCL/RCCL")
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        elif requested == "cpu":
            if args.distributed_backend != "gloo":
                raise RuntimeError("CPU DDP requires Gloo")
            args.device = torch.device("cpu")
        else:
            raise ValueError("unsupported distributed device")
        dist.init_process_group(
            backend=args.distributed_backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
    else:
        args.device = torch.device(requested)
        if args.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/ROCm requested but unavailable")
            torch.cuda.set_device(args.device)
    validate_args(args)


def cleanup_distributed(args: argparse.Namespace) -> None:
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args: argparse.Namespace) -> bool:
    return args.rank == 0


def barrier(args: argparse.Namespace) -> None:
    if args.distributed:
        if args.device.type == "cuda":
            dist.barrier(device_ids=[args.local_rank])
        else:
            dist.barrier()


def run_on_rank_zero(args: argparse.Namespace, operation: Callable[[], Any]) -> Any:
    barrier(args)
    result: Any = None
    failure: str | None = None
    if is_main_process(args):
        try:
            result = operation()
        except Exception:
            failure = traceback.format_exc()
    if args.distributed:
        payload = [failure, result]
        dist.broadcast_object_list(payload, src=0)
        failure, result = payload
    if failure is not None:
        raise RuntimeError(f"rank-zero operation failed:\n{failure}")
    barrier(args)
    return result


def gather_rank_objects(local: Any, args: argparse.Namespace) -> list[Any]:
    if not args.distributed:
        return [local]
    gathered: list[Any] = [None] * args.world_size
    dist.all_gather_object(gathered, local)
    if any(value is None for value in gathered):
        raise RuntimeError("failed to gather all rank objects")
    return gathered


def seed_parameters(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def make_prior_generator(args: argparse.Namespace) -> torch.Generator:
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed + 1_000_003 * args.rank + 17)
    return generator


def make_bank_generator(args: argparse.Namespace) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(args.seed + 29)


class GlobalBankStream:
    """Identical deterministic dataset permutation on every rank."""

    def __init__(self, dataset: CelebA64PixelDataset, *, batch_size: int, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.batches_per_epoch = len(dataset) // self.batch_size
        if self.batches_per_epoch < 1:
            raise ValueError("training split is smaller than one bank-push batch")
        self.usable_count = self.batches_per_epoch * self.batch_size
        self.epoch = 0
        self.batch_offset = 0
        self._permutation = self._make_permutation()

    def _make_permutation(self) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        return torch.randperm(len(self.dataset), generator=generator)[: self.usable_count]

    def next_indices(self) -> Tensor:
        if self.batch_offset == self.batches_per_epoch:
            self.epoch += 1
            self.batch_offset = 0
            self._permutation = self._make_permutation()
        start = self.batch_offset * self.batch_size
        indices = self._permutation[start : start + self.batch_size]
        self.batch_offset += 1
        return indices

    def next(self) -> Tensor:
        return self.dataset.batch(self.next_indices())

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_offset": self.batch_offset}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"epoch", "batch_offset"}:
            raise ValueError("malformed bank stream state")
        epoch, offset = int(state["epoch"]), int(state["batch_offset"])
        if epoch < 0 or not 0 <= offset <= self.batches_per_epoch:
            raise ValueError("invalid bank stream position")
        self.epoch, self.batch_offset = epoch, offset
        self._permutation = self._make_permutation()


def augment_bank_images(
    images_uint8: Tensor,
    mode: str,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Map aligned uint8 files to FP32 [-1,1] with an explicit CPU RNG."""

    if images_uint8.device.type != "cpu" or images_uint8.dtype != torch.uint8:
        raise ValueError("bank augmentation expects CPU uint8 images")
    if mode not in AUGMENTATIONS:
        raise ValueError("unsupported bank augmentation")
    images = images_uint8.float().div_(255.0)
    if mode == "aligned_flip":
        flip = torch.rand(images.shape[0], generator=generator) < 0.5
        images = torch.where(flip[:, None, None, None], images.flip(-1), images)
    return images.mul_(2.0).sub_(1.0)


def learning_rate_at_step(step_index: int, args: argparse.Namespace) -> float:
    if args.lr_warmup_steps and step_index < args.lr_warmup_steps:
        progress = step_index / args.lr_warmup_steps
        return args.lr_warmup_start + progress * (args.lr - args.lr_warmup_start)
    return args.lr


def make_generator(args: argparse.Namespace) -> CelebA64DirectPixelGMDDiT:
    return CelebA64DirectPixelGMDDiT(
        patch_size=args.patch_size,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        noise_classes=args.noise_classes,
        noise_coords=args.noise_coords,
        condition_tokens=args.condition_tokens,
        gradient_checkpointing=args.gradient_checkpointing,
        intended_inference_precision=(
            "bfloat16_autocast" if args.use_bfloat16 else "float32"
        ),
    )


def unwrap_generator(model: nn.Module) -> CelebA64DirectPixelGMDDiT:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(unwrapped, CelebA64DirectPixelGMDDiT):
        raise TypeError("unexpected generator model")
    return unwrapped


def make_ema(model: CelebA64DirectPixelGMDDiT) -> CelebA64DirectPixelGMDDiT:
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(
    ema: CelebA64DirectPixelGMDDiT,
    model: CelebA64DirectPixelGMDDiT,
    decay: float,
) -> None:
    for ema_parameter, parameter in zip(ema.parameters(), model.parameters(), strict=True):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def _source_hashes() -> dict[str, str]:
    missing = [relative for relative in SOURCE_FILES if not (REPO_ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"required source files are missing: {missing}")
    return {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCE_FILES}


def verify_launcher_source_manifest(
    args: argparse.Namespace,
) -> dict[str, str] | None:
    path = args.expected_source_manifest
    expected_digest = args.expected_source_manifest_sha256
    if path is None:
        return None
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("expected source manifest must be a regular file")
    if not _valid_sha256(expected_digest) or sha256_file(path) != expected_digest:
        raise ValueError("expected source manifest external SHA-256 mismatch")
    expected_paths = [*SOURCE_FILES, str(args.feature_checkpoint)]
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != len(expected_paths):
        raise ValueError("expected source manifest entry count differs")
    parsed: dict[str, str] = {}
    for line, expected_path in zip(lines, expected_paths, strict=True):
        digest, separator, observed_path = line.partition("  ")
        if separator != "  " or not _valid_sha256(digest) or observed_path != expected_path:
            raise ValueError("expected source manifest entry differs")
        parsed[expected_path] = digest
    expected_sources = {relative: parsed[relative] for relative in SOURCE_FILES}
    if parsed[str(args.feature_checkpoint)] != args.feature_checkpoint_sha256:
        raise ValueError("source manifest feature artifact SHA-256 differs")
    if _source_hashes() != expected_sources:
        raise RuntimeError("training sources differ from the pre-import launcher manifest")
    return expected_sources


def archive_source_snapshot(
    run_dir: Path,
    name: str,
    expected_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    destination_root = run_dir / name
    if destination_root.exists():
        raise FileExistsError(f"source snapshot already exists: {destination_root}")
    before = _source_hashes()
    for relative in SOURCE_FILES:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    after = _source_hashes()
    copied = {
        relative: sha256_file(destination_root / relative) for relative in SOURCE_FILES
    }
    if before != after or before != copied:
        raise RuntimeError("training sources changed while snapshotting")
    if expected_hashes is not None and before != dict(expected_hashes):
        raise RuntimeError("source snapshot differs from the launcher manifest")
    manifest = {
        "format": SOURCE_SNAPSHOT_FORMAT,
        "version": SOURCE_SNAPSHOT_VERSION,
        "created_at_utc": utc_now(),
        "sha256": copied,
    }
    path = atomic_json_dump(destination_root / "manifest.json", manifest)
    return {**manifest, "manifest": str(path.resolve()), "manifest_sha256": sha256_file(path)}


def _git_provenance() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *arguments], cwd=REPO_ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run("status", "--short"),
    }


def feature_checkpoint_identity(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(args.feature_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("feature checkpoint payload is not a mapping")
    architecture = payload.get("architecture")
    schema = payload.get("feature_schema")
    metadata = payload.get("metadata")
    integrity = payload.get("integrity")
    if not all(
        isinstance(value, Mapping)
        for value in (architecture, schema, metadata, integrity)
    ):
        raise ValueError(
            "feature checkpoint lacks strict architecture/schema/metadata/integrity"
        )
    required_metadata = {
        "completed_step",
        "training_signature",
        "training_signature_sha256",
        "weight_kind",
    }
    if not required_metadata.issubset(metadata):
        raise ValueError(
            "feature checkpoint is not an exported phase-2 training artifact"
        )
    completed_step = metadata["completed_step"]
    feature_training_signature = metadata["training_signature"]
    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise ValueError("feature checkpoint completed_step is malformed")
    if completed_step < 1:
        raise ValueError("feature checkpoint has no completed phase-2 update")
    if not isinstance(feature_training_signature, Mapping):
        raise ValueError("feature checkpoint phase-2 signature is malformed")
    if (
        feature_training_signature.get("objective")
        != "masked_only_packed_pixel_reconstruction"
    ):
        raise ValueError(
            "feature checkpoint was not trained with the phase-2 MAE objective"
        )
    if metadata["training_signature_sha256"] != canonical_json_sha256(
        dict(feature_training_signature)
    ):
        raise ValueError(
            "feature checkpoint phase-2 training signature SHA-256 mismatch"
        )
    expected_training_signature = args.expected_feature_training_signature_sha256
    if not args.smoke and not _valid_sha256(expected_training_signature):
        raise ValueError(
            "non-smoke feature identity requires the expected phase-2 training signature SHA-256"
        )
    if (
        expected_training_signature is not None
        and metadata["training_signature_sha256"] != expected_training_signature
    ):
        raise ValueError(
            "feature checkpoint phase-2 training signature differs from the pinned expectation"
        )
    if metadata["weight_kind"] != args.feature_weight_kind:
        raise ValueError(
            f"feature checkpoint weight_kind={metadata['weight_kind']!r}, "
            f"expected {args.feature_weight_kind!r}"
        )
    if metadata["weight_kind"] != "ema":
        raise ValueError("the production pixel-GMD contract requires EMA features")
    production_architecture = PRODUCTION_FEATURE_ARCHITECTURE
    architecture_mismatches = {
        key: {"expected": expected, "observed": architecture.get(key)}
        for key, expected in production_architecture.items()
        if architecture.get(key) != expected
    }
    if architecture_mismatches and not args.smoke:
        raise ValueError(
            "feature checkpoint violates the production MAE architecture contract: "
            f"{architecture_mismatches}"
        )
    short_checkpoint = completed_step < PRODUCTION_FEATURE_MINIMUM_UPDATES
    if short_checkpoint and not (
        args.smoke or args.allow_short_feature_checkpoint_preflight
    ):
        raise ValueError(
            "feature checkpoint is shorter than the production 10000-update floor; "
            "only an explicit <=2-step GPU preflight may relax it"
        )
    return {
        "file_sha256": args.feature_checkpoint_sha256,
        "weight_kind": metadata["weight_kind"],
        "completed_step": completed_step,
        "phase2_training_signature_sha256": metadata[
            "training_signature_sha256"
        ],
        "expected_phase2_training_signature_sha256": expected_training_signature,
        "architecture": dict(architecture),
        "feature_schema": dict(schema),
        "state_dict_sha256": integrity.get("state_dict_sha256"),
        "production_contract": {
            "weight_kind": "ema",
            **production_architecture,
            "minimum_completed_step": PRODUCTION_FEATURE_MINIMUM_UPDATES,
            "architecture_satisfied": not architecture_mismatches,
            "updates_satisfied": not short_checkpoint,
            "short_checkpoint_preflight_opt_in": bool(
                args.allow_short_feature_checkpoint_preflight
            ),
            "unit_smoke_exception": bool(args.smoke),
        },
    }


def training_signature(
    args: argparse.Namespace,
    dataset_manifest: Mapping[str, Any],
    generator: CelebA64DirectPixelGMDDiT,
    feature_identity: Mapping[str, Any],
) -> dict[str, Any]:
    source_hashes = _source_hashes()
    expected_source_hashes = args.launcher_expected_source_hashes
    if (
        expected_source_hashes is not None
        and source_hashes != expected_source_hashes
    ):
        raise RuntimeError("training sources changed after launcher-manifest verification")
    return {
        "version": TRAINING_SIGNATURE_VERSION,
        "objective": "released_normalized_multitemperature_GMD",
        "objective_implementation": "src.official_drifting.official_drifting_loss",
        "feature_to_groups": "src.official_drifting.token_features_to_groups",
        "generator": generator.architecture_config(),
        "feature_checkpoint": dict(feature_identity),
        "temperatures_in_accumulation_order": list(args.temperatures),
        "support": {
            "generated_global": args.global_generated,
            "generated_local": args.local_generated,
            "generated_repulsion": "detached generated support internal to official loss",
            "positive_count": args.positive_count,
            "positive_bank_size": args.positive_bank_size,
            "fixed_negative_count": args.negative_count,
            "fixed_negative_bank_size": args.negative_bank_size,
            "bank_push_per_update": args.bank_push,
            "fixed_negative": (
                "real examples from a global bank; included in row/column "
                "affinity softmax with exactly zero post-softmax weight at CFG=1"
            ),
            "fixed_negative_weight": 0.0,
            "cross_rank_generated": "differentiable all_gather before frozen features and GMD",
            "feature_extraction_distribution": (
                "global generated/positive/negative supports are encoded on every rank; "
                "correct but replicated world_size times pending measured full-size smoke"
            ),
        },
        "real_model_space": "RGB NCHW float32 [-1,1]",
        "generator_training_output": "direct unbounded RGB; no tanh or clip before GMD",
        "sampling_postprocess": "clip to [-1,1] only outside training",
        "augmentation": args.augmentation,
        "optimizer": "torch.optim.AdamW",
        "learning_rate": args.lr,
        "lr_warmup_start": args.lr_warmup_start,
        "lr_warmup_steps": args.lr_warmup_steps,
        "schedule": "linear Optax-style warmup then constant",
        "schedule_reference": {
            "released_total_updates": 100_000,
            "released_warmup_updates": 10_000,
            "absolute_warmup_matches_release": args.lr_warmup_steps == 10_000,
            "interpretation": (
                "mechanical released absolute warmup"
                if args.lr_warmup_steps == 10_000
                else "compressed diagnostic warmup; not mechanical schedule fidelity"
            ),
        },
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "ema_initialized_from": "theta_0",
        "use_bfloat16": args.use_bfloat16,
        "precision": (
            "CUDA/ROCm autocast BF16; FP32 params, optimizer, GMD geometry"
            if args.use_bfloat16
            else "float32 generator, feature extraction, optimizer, and GMD geometry"
        ),
        "inference_precision_contract": generator.intended_inference_precision,
        "world_size": args.world_size,
        "ddp_broadcast_buffers": False,
        "ddp_static_graph": True,
        "device_type": args.device.type,
        "accelerator_name": (
            torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else "cpu"
        ),
        "distributed_backend": args.distributed_backend,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "pillow_version": str(PIL_VERSION),
        "numpy_version": str(np.__version__),
        "seed": args.seed,
        "deterministic_algorithms": args.deterministic,
        "detailed_gradient_verification": {
            "enabled": args.verify_gradients,
            "global_step_1": args.verify_gradients,
            "interval_after_step_1": args.gradient_verification_interval,
            "clip_grad_norm_nonfinite_check_every_update": True,
        },
        "preload_uint8": args.preload_uint8,
        "dataset_root": dataset_manifest["root"],
        "dataset_files_sha256": dataset_manifest["files_sha256"],
        "dataset_image_count": dataset_manifest["image_count"],
        "train_start_index": dataset_manifest["train_start_index"],
        "train_end_index": dataset_manifest["train_end_index"],
        "source_files_sha256": source_hashes,
        "launcher_expected_source_manifest_sha256": (
            args.expected_source_manifest_sha256
        ),
        "official_release_commit": OFFICIAL_RELEASE_COMMIT,
    }


def resolve_run_directory(args: argparse.Namespace) -> Path:
    def resolve() -> str:
        if args.resume is not None:
            parent = args.resume.resolve().parent
            inferred = parent.parent if parent.name == "checkpoints" else parent
            if args.run_dir is not None and args.run_dir.resolve() != inferred:
                raise ValueError("--run-dir differs from directory implied by --resume")
            if not inferred.is_dir():
                raise FileNotFoundError("resume run directory is absent")
            path = inferred
        elif args.run_dir is not None:
            path = args.run_dir.resolve()
            if path.exists() and any(path.iterdir()):
                raise FileExistsError("new run directory is not empty")
            path.mkdir(parents=True, exist_ok=True)
        else:
            label = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in args.run_label
            ).strip("_")
            path = (
                args.output_root / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{label}"
            ).resolve()
            path.mkdir(parents=True, exist_ok=False)
        return str(path)

    return Path(run_on_rank_zero(args, resolve))


def reduce_stats(stats: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Tensor]:
    keys = list(stats)
    packed = torch.stack(
        [
            value.detach().to(args.device, dtype=torch.float32).reshape(())
            if torch.is_tensor(value)
            else torch.tensor(float(value), device=args.device)
            for value in stats.values()
        ]
    )
    if args.distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(args.world_size)
    return {name: packed[index] for index, name in enumerate(keys)}


def reduce_max_stats(
    stats: Mapping[str, Any], args: argparse.Namespace
) -> dict[str, Tensor]:
    """Return scalar maxima across ranks for timing and memory telemetry."""

    keys = list(stats)
    packed = torch.stack(
        [
            value.detach().to(args.device, dtype=torch.float64).reshape(())
            if torch.is_tensor(value)
            else torch.tensor(float(value), device=args.device, dtype=torch.float64)
            for value in stats.values()
        ]
    )
    if args.distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.MAX)
    return {name: packed[index] for index, name in enumerate(keys)}


def compact_gmd_statistics(statistics: Mapping[str, Tensor]) -> dict[str, Tensor]:
    scales = [value for key, value in statistics.items() if key.startswith("scale/")]
    energies = [value for key, value in statistics.items() if key.startswith("force_energy_")]
    family_losses = [value for key, value in statistics.items() if key.startswith("loss/")]
    if not scales or not energies or not family_losses:
        raise RuntimeError("GMD feature statistics are incomplete")
    return {
        "gmd_loss": statistics["loss"],
        "feature_family_count": float(len(family_losses)),
        "family_loss_mean": torch.stack(family_losses).mean(),
        "family_loss_max": torch.stack(family_losses).max(),
        "feature_scale_mean": torch.stack(scales).mean(),
        "force_energy_mean": torch.stack(energies).mean(),
        "generated_support": statistics["support/generated_global"],
        "positive_support": statistics["support/positive"],
        "fixed_negative_support": statistics["support/fixed_negative"],
        "fixed_negative_weight_sum": statistics[
            "support/fixed_negative_weight_sum"
        ],
    }


def verify_gradients(
    model: nn.Module,
    feature_encoder: nn.Module,
    args: argparse.Namespace,
) -> None:
    if any(parameter.grad is not None for parameter in feature_encoder.parameters()):
        raise RuntimeError("frozen feature encoder received parameter gradients")
    if not args.verify_gradients:
        return
    gradients = [parameter.grad for parameter in model.parameters()]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("at least one generator parameter has no gradient")
    assert all(gradient is not None for gradient in gradients)
    all_finite = torch.stack(
        [torch.isfinite(gradient).all() for gradient in gradients]
    ).all().to(dtype=torch.int32)
    if args.distributed:
        dist.all_reduce(all_finite, op=dist.ReduceOp.MIN)
    if not bool(all_finite):
        raise FloatingPointError("non-finite generator gradient")
    fingerprint = torch.stack(
        (
            sum(gradient.float().sum() for gradient in gradients),
            sum(gradient.float().square().sum() for gradient in gradients),
            torch.stack([gradient.float().abs().max() for gradient in gradients]).max(),
        )
    )
    if args.distributed:
        gathered = [torch.empty_like(fingerprint) for _ in range(args.world_size)]
        dist.all_gather(gathered, fingerprint)
        if not all(torch.equal(gathered[0], value) for value in gathered[1:]):
            raise RuntimeError("DDP generator gradient fingerprints differ")


def _checkpoint_integrity(
    *,
    model_state: Mapping[str, Tensor],
    ema_state: Mapping[str, Tensor],
    optimizer_state: Mapping[str, Any],
    stream_state: Mapping[str, Any],
    positive_bank_state: Mapping[str, Any],
    negative_bank_state: Mapping[str, Any],
    bank_rng_state: Tensor,
    prior_rng_states: Sequence[Any],
    signature: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "model_state_sha256": state_dict_sha256(model_state),
        "ema_state_sha256": state_dict_sha256(ema_state),
        "optimizer_state_sha256": torch_tree_sha256(optimizer_state),
        "stream_state_sha256": torch_tree_sha256(stream_state),
        "positive_bank_state_sha256": torch_tree_sha256(positive_bank_state),
        "negative_bank_state_sha256": torch_tree_sha256(negative_bank_state),
        "bank_rng_state_sha256": torch_tree_sha256(bank_rng_state),
        "prior_rng_states_sha256": torch_tree_sha256(list(prior_rng_states)),
        "training_signature_sha256": canonical_json_sha256(dict(signature)),
    }


def save_checkpoint(
    path: Path,
    *,
    completed_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: CelebA64DirectPixelGMDDiT,
    stream: GlobalBankStream,
    positive_bank: PositiveImageBank,
    negative_bank: NegativeImageBank,
    bank_generator: torch.Generator,
    prior_generator: torch.Generator,
    signature: Mapping[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> str:
    prior_rng_states = gather_rank_objects(prior_generator.get_state().cpu(), args)
    stream_states = gather_rank_objects(stream.state_dict(), args)
    positive_bank_states = gather_rank_objects(positive_bank.state_dict(), args)
    negative_bank_states = gather_rank_objects(negative_bank.state_dict(), args)
    bank_rng_states = gather_rank_objects(bank_generator.get_state().cpu(), args)
    stream_hashes = [torch_tree_sha256(value) for value in stream_states]
    positive_bank_hashes = [
        torch_tree_sha256(value) for value in positive_bank_states
    ]
    negative_bank_hashes = [
        torch_tree_sha256(value) for value in negative_bank_states
    ]
    bank_rng_hashes = [torch_tree_sha256(value) for value in bank_rng_states]
    if (
        len(set(stream_hashes)) != 1
        or len(set(positive_bank_hashes)) != 1
        or len(set(negative_bank_hashes)) != 1
        or len(set(bank_rng_hashes)) != 1
    ):
        raise RuntimeError("global bank/support state diverged across ranks")

    def save() -> str:
        model_state = {name: value.detach().cpu() for name, value in unwrap_generator(model).state_dict().items()}
        ema_state = {name: value.detach().cpu() for name, value in ema.state_dict().items()}
        optimizer_state = optimizer.state_dict()
        stream_state = stream_states[0]
        positive_bank_state = positive_bank_states[0]
        negative_bank_state = negative_bank_states[0]
        bank_rng_state = bank_rng_states[0]
        payload = {
            "format": TRAINING_CHECKPOINT_FORMAT,
            "version": TRAINING_CHECKPOINT_VERSION,
            "completed_step": completed_step,
            "model": model_state,
            "ema": ema_state,
            "optimizer": optimizer_state,
            "data_stream": stream_state,
            "positive_bank": positive_bank_state,
            "negative_bank": negative_bank_state,
            "bank_rng_state": bank_rng_state,
            "prior_rng_by_rank": prior_rng_states,
            "training_signature": dict(signature),
            "run_dir": str(run_dir.resolve()),
            "saved_at_utc": utc_now(),
            "integrity": _checkpoint_integrity(
                model_state=model_state,
                ema_state=ema_state,
                optimizer_state=optimizer_state,
                stream_state=stream_state,
                positive_bank_state=positive_bank_state,
                negative_bank_state=negative_bank_state,
                bank_rng_state=bank_rng_state,
                prior_rng_states=prior_rng_states,
                signature=signature,
            ),
        }
        atomic_torch_save(path, payload)
        digest = sha256_file(path)
        atomic_text_write(Path(f"{path}.sha256"), f"{digest}  {path.name}\n")
        return digest

    return str(run_on_rank_zero(args, save))


def _verified_checkpoint_file(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("resume checkpoint must be a regular file")
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise FileNotFoundError("resume checkpoint sidecar is absent")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name or not _valid_sha256(fields[0]):
        raise ValueError("resume checkpoint sidecar is malformed")
    observed = sha256_file(path)
    if observed != fields[0] or observed != expected_sha256:
        raise ValueError("resume checkpoint external SHA-256 mismatch")


def _validate_checkpoint_integrity(checkpoint: Mapping[str, Any]) -> None:
    integrity = checkpoint.get("integrity")
    expected_fields = {
        "algorithm",
        "model_state_sha256",
        "ema_state_sha256",
        "optimizer_state_sha256",
        "stream_state_sha256",
        "positive_bank_state_sha256",
        "negative_bank_state_sha256",
        "bank_rng_state_sha256",
        "prior_rng_states_sha256",
        "training_signature_sha256",
    }
    if not isinstance(integrity, Mapping) or set(integrity) != expected_fields:
        raise ValueError("resume checkpoint integrity schema differs")
    calculated = _checkpoint_integrity(
        model_state=checkpoint["model"],
        ema_state=checkpoint["ema"],
        optimizer_state=checkpoint["optimizer"],
        stream_state=checkpoint["data_stream"],
        positive_bank_state=checkpoint["positive_bank"],
        negative_bank_state=checkpoint["negative_bank"],
        bank_rng_state=checkpoint["bank_rng_state"],
        prior_rng_states=checkpoint["prior_rng_by_rank"],
        signature=checkpoint["training_signature"],
    )
    if dict(integrity) != calculated:
        differing = sorted(key for key in expected_fields if integrity.get(key) != calculated.get(key))
        raise ValueError(f"resume checkpoint internal SHA-256 differs for {differing}")


def load_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: CelebA64DirectPixelGMDDiT,
    stream: GlobalBankStream,
    positive_bank: PositiveImageBank,
    negative_bank: NegativeImageBank,
    bank_generator: torch.Generator,
    prior_generator: torch.Generator,
    signature: Mapping[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> int:
    _verified_checkpoint_file(path, expected_sha256)
    checkpoint = torch.load(path, map_location=args.device, weights_only=True)
    expected_top = {
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
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != expected_top:
        raise ValueError("resume expects the exact direct-pixel GMD schema")
    if checkpoint["format"] != TRAINING_CHECKPOINT_FORMAT or checkpoint["version"] != TRAINING_CHECKPOINT_VERSION:
        raise ValueError("unsupported direct-pixel GMD checkpoint")
    _validate_checkpoint_integrity(checkpoint)
    if checkpoint["training_signature"] != dict(signature):
        found = checkpoint["training_signature"] if isinstance(checkpoint["training_signature"], Mapping) else {}
        differing = sorted(key for key in set(found).union(signature) if found.get(key) != signature.get(key))
        raise ValueError(f"resume training signature differs for {differing}")
    if Path(str(checkpoint["run_dir"])).resolve() != run_dir.resolve():
        raise ValueError("resume run-directory provenance differs")
    unwrap_generator(model).load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    stream.load_state_dict(checkpoint["data_stream"])
    positive_bank.load_state_dict(checkpoint["positive_bank"])
    negative_bank.load_state_dict(checkpoint["negative_bank"])
    bank_generator.set_state(checkpoint["bank_rng_state"].cpu())
    prior_states = checkpoint["prior_rng_by_rank"]
    if not isinstance(prior_states, list) or len(prior_states) != args.world_size:
        raise ValueError("resume prior RNG world size differs")
    prior_generator.set_state(prior_states[args.rank].cpu())
    completed = int(checkpoint["completed_step"])
    if completed < 0:
        raise ValueError("negative completed step")
    return completed


def write_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    dataset_manifest: Mapping[str, Any],
    signature: Mapping[str, Any],
    generator: CelebA64DirectPixelGMDDiT,
) -> None:
    resume_token = "" if args.resume is None else f"_resume_{utc_now().replace(':', '-').replace('+', '_')}"
    snapshot = archive_source_snapshot(
        run_dir,
        f"source_snapshot{resume_token}",
        args.launcher_expected_source_hashes,
    )
    dataset_path = atomic_json_dump(run_dir / f"dataset_manifest{resume_token}.json", dataset_manifest)
    config = {
        **{key: value for key, value in vars(args).items() if not key.startswith("_")},
        "feature_checkpoint_sha256": args.feature_checkpoint_sha256,
        "parameter_count": sum(parameter.numel() for parameter in generator.parameters()),
        "training_signature": signature,
        "requested_total_steps_not_in_resume_signature": args.steps,
    }
    atomic_json_dump(run_dir / f"config{resume_token}.json", _json_safe(config))
    provenance = {
        "created_at_utc": utc_now(),
        "command": args.command,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "pillow": str(PIL_VERSION),
        "numpy": str(np.__version__),
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": getattr(torch.version, "hip", None),
        "world_size": args.world_size,
        "git": _git_provenance(),
        "source_snapshot": snapshot,
        "dataset_manifest": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "files_sha256": dataset_manifest["files_sha256"],
        },
        "official_pixel_generator_config": OFFICIAL_PIXEL_GENERATOR_CONFIG,
        "adaptations": {
            "data": "aligned unconditional CelebA64 instead of class-conditional ImageNet256",
            "patch": f"{args.patch_size}px gives {64 // args.patch_size}x{64 // args.patch_size} tokens; official 16px gives 16x16",
            "conditioning": (
                "class lookup and CFG extrapolation removed; unconditional and "
                "discrete noise embeddings plus the trainable released "
                "TimestepEmbedder/RMSNorm branch at cfg_scale=1 are retained"
            ),
            "support": (
                "one unconditional positive bank plus one global real-example negative "
                "bank; at CFG=1 negative weights are exactly zero after affinities, "
                "while those samples remain in both softmax denominators"
            ),
            "feature_compute": (
                "the complete generated, positive, and fixed-negative feature extraction "
                "is currently replicated on every DDP rank; retain until a measured "
                "full-size BF16/RCCL smoke validates memory and update time"
            ),
            "augmentation": f"{args.augmentation}; release RandomResizedCrop deliberately omitted for registered faces",
            "framework": "PyTorch DDP on one 8-GPU node instead of JAX HSDP",
        },
        "explicit_nonclaims": [
            "No FID is computed or inferred by this trainer.",
            "This is not a VAE or latent generator.",
            "This is not the historical RGB KDE/WGF objective.",
        ],
    }
    atomic_json_dump(run_dir / f"provenance{resume_token}.json", provenance)


def export_artifacts(
    run_dir: Path,
    *,
    completed_step: int,
    raw: CelebA64DirectPixelGMDDiT,
    ema: CelebA64DirectPixelGMDDiT,
    signature: Mapping[str, Any],
    final_checkpoint: Path,
    final_checkpoint_sha256: str,
) -> dict[str, Any]:
    common = {
        "completed_step": completed_step,
        "training_signature_sha256": canonical_json_sha256(dict(signature)),
        "feature_checkpoint_sha256": signature["feature_checkpoint"]["file_sha256"],
        "objective": signature["objective"],
        "intended_inference_precision": signature[
            "inference_precision_contract"
        ],
        "created_at_utc": utc_now(),
    }
    raw_path = run_dir / "generator_raw_final.pt"
    ema_path = run_dir / "generator_ema_final.pt"
    atomic_torch_save(raw_path, generator_artifact_dict(raw, metadata={**common, "weight_kind": "raw"}))
    atomic_torch_save(
        ema_path,
        generator_artifact_dict(
            ema,
            metadata={
                **common,
                "weight_kind": "ema",
                "ema_decay": signature["ema_decay"],
                "ema_initialized_from": "theta_0",
            },
        ),
    )
    load_pixel_gmd_generator_artifact(raw_path, "cpu")
    load_pixel_gmd_generator_artifact(ema_path, "cpu")
    manifest = {
        "format": ARTIFACT_MANIFEST_FORMAT,
        "version": ARTIFACT_MANIFEST_VERSION,
        "created_at_utc": utc_now(),
        "completed_step": completed_step,
        "strict_generator_reload": True,
        "intended_inference_precision": signature[
            "inference_precision_contract"
        ],
        "training_signature_sha256": canonical_json_sha256(dict(signature)),
        "artifacts": {
            "training_checkpoint": {"path": str(final_checkpoint.resolve()), "sha256": final_checkpoint_sha256},
            "raw_generator": {"path": str(raw_path.resolve()), "sha256": sha256_file(raw_path)},
            "ema_generator": {"path": str(ema_path.resolve()), "sha256": sha256_file(ema_path)},
        },
    }
    path = atomic_json_dump(run_dir / "artifact_manifest.json", manifest)
    atomic_text_write(run_dir / "artifact_manifest.json.sha256", f"{sha256_file(path)}  {path.name}\n")
    return manifest


def train(args: argparse.Namespace) -> Path:
    dataset_manifest = run_on_rank_zero(
        args,
        lambda: build_dataset_manifest(
            args.data_dir.resolve(), args.train_start_index, args.train_end_index
        ),
    )
    if not isinstance(dataset_manifest, Mapping):
        raise RuntimeError("dataset manifest broadcast failed")
    dataset = CelebA64PixelDataset(
        args.data_dir.resolve(), dataset_manifest, preload_uint8=args.preload_uint8
    )
    verified_manifest = run_on_rank_zero(
        args,
        lambda: build_dataset_manifest(
            args.data_dir.resolve(), args.train_start_index, args.train_end_index
        ),
    )
    if verified_manifest != dataset_manifest:
        raise RuntimeError("CelebA dataset changed while being preloaded")
    stream = GlobalBankStream(dataset, batch_size=args.bank_push, seed=args.seed + 101)
    positive_bank = PositiveImageBank(args.positive_bank_size)
    negative_bank = NegativeImageBank(args.negative_bank_size)
    bank_generator = make_bank_generator(args)

    # Strict feature loading verifies internal hashes and freezes every weight
    # while intentionally preserving the Jacobian with respect to RGB input.
    feature_identity = feature_checkpoint_identity(args)
    feature_encoder = load_frozen_pixel_feature_encoder(args.feature_checkpoint, args.device)

    seed_parameters(args)
    generator = make_generator(args).to(args.device)
    train_model: nn.Module = generator
    if args.distributed:
        ddp_options: dict[str, Any] = {
            "broadcast_buffers": False,
            # The graph has no data-dependent parameter usage.  Fixing DDP's
            # buckets avoids a first-backward rebuild after process restart,
            # which otherwise breaks bitwise resume despite identical losses.
            "static_graph": True,
        }
        if args.device.type == "cuda":
            ddp_options.update(device_ids=[args.local_rank], output_device=args.local_rank)
        train_model = DistributedDataParallel(generator, **ddp_options)
    prior_generator = make_prior_generator(args)
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    ema = make_ema(unwrap_generator(train_model))
    signature = training_signature(
        args, dataset_manifest, unwrap_generator(train_model), feature_identity
    )
    run_dir = resolve_run_directory(args)
    start_step = 0
    if args.resume is not None:
        assert args.resume_sha256 is not None
        start_step = load_checkpoint(
            args.resume,
            expected_sha256=args.resume_sha256,
            model=train_model,
            optimizer=optimizer,
            ema=ema,
            stream=stream,
            positive_bank=positive_bank,
            negative_bank=negative_bank,
            bank_generator=bank_generator,
            prior_generator=prior_generator,
            signature=signature,
            run_dir=run_dir,
            args=args,
        )
    if start_step >= args.steps:
        raise ValueError("--steps must exceed the resumed completed step")

    run_on_rank_zero(
        args,
        lambda: write_metadata(
            args, run_dir, dataset_manifest, signature, unwrap_generator(train_model)
        ),
    )
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    if is_main_process(args):
        logger.log(
            {
                "event": "resume" if start_step else "start",
                "step": start_step,
                "world_size": args.world_size,
                "global_generated": args.global_generated,
                "positive_count": args.positive_count,
                "negative_count": args.negative_count,
                "feature_checkpoint_sha256": args.feature_checkpoint_sha256,
                "dataset_files_sha256": dataset_manifest["files_sha256"],
                "parameter_count": sum(
                    parameter.numel() for parameter in unwrap_generator(train_model).parameters()
                ),
            }
        )

    if args.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.monotonic()
    progress = trange(
        start_step,
        args.steps,
        initial=start_step,
        total=args.steps,
        desc="celeba64:direct-pixel-gmd",
        disable=not is_main_process(args),
    )
    for step_index in progress:
        update_started = time.monotonic()
        learning_rate = learning_rate_at_step(step_index, args)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        pushed = augment_bank_images(
            stream.next(), args.augmentation, generator=bank_generator
        )
        # Both supports consume exactly the same real batch, matching the two
        # memory-bank insertions in the released trainer.  Sampling order and
        # the single CPU RNG are causal checkpoint state.
        positive_bank.add(pushed)
        negative_bank.add(pushed)
        positives = positive_bank.sample(
            args.positive_count, generator=bank_generator
        ).to(
            args.device, non_blocking=args.device.type == "cuda"
        )
        fixed_negatives = negative_bank.sample(
            args.negative_count, generator=bank_generator
        ).to(args.device, non_blocking=args.device.type == "cuda")
        prior, noise_labels = unwrap_generator(train_model).sample_prior(
            args.local_generated, device=args.device, generator=prior_generator
        )

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = args.use_bfloat16 and args.device.type == "cuda"
        with torch.autocast(
            device_type=args.device.type, dtype=torch.bfloat16, enabled=autocast_enabled
        ):
            local_generated = train_model(prior, noise_labels)
            loss, detailed_statistics = global_pixel_gmd_loss(
                feature_encoder,
                local_generated,
                positives,
                fixed_negatives,
                temperatures=args.temperatures,
            )
        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        if args.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"non-finite GMD loss at step {step_index + 1}")
        loss.backward()
        completed_step = step_index + 1
        detailed_gradient_check = args.verify_gradients and (
            completed_step == 1
            or (
                args.gradient_verification_interval > 0
                and completed_step % args.gradient_verification_interval == 0
            )
        )
        if detailed_gradient_check:
            verify_gradients(train_model, feature_encoder, args)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            train_model.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        update_ema(ema, unwrap_generator(train_model), args.ema_decay)

        should_log = (
            completed_step == 1
            or completed_step == args.steps
            or completed_step % args.log_every == 0
        )
        if should_log:
            if args.device.type == "cuda":
                torch.cuda.synchronize(args.device)
            log_time = time.monotonic()
            maxima = {
                "seconds_per_update_max_rank": log_time - update_started
            }
            if args.device.type == "cuda":
                maxima.update(
                    {
                        "gpu_peak_memory_allocated_bytes_max_rank": torch.cuda.max_memory_allocated(
                            args.device
                        ),
                        "gpu_peak_memory_reserved_bytes_max_rank": torch.cuda.max_memory_reserved(
                            args.device
                        ),
                    }
                )
            reduced_maxima = reduce_max_stats(maxima, args)
            compact = compact_gmd_statistics(detailed_statistics)
            reduced = reduce_stats(
                {
                    **compact,
                    "prior_mean": prior.float().mean(),
                    "prior_std": prior.float().std(correction=0),
                    "generated_mean": local_generated.detach().float().mean(),
                    "generated_std": local_generated.detach().float().std(correction=0),
                    "positive_mean": positives.float().mean(),
                    "positive_std": positives.float().std(correction=0),
                    "fixed_negative_mean": fixed_negatives.float().mean(),
                    "fixed_negative_std": fixed_negatives.float().std(correction=0),
                    "gradient_norm_preclip": gradient_norm,
                },
                args,
            )
            record = {
                "event": "train",
                "step": completed_step,
                "epoch": stream.epoch,
                "batch_offset": stream.batch_offset,
                "positive_bank_count": positive_bank.count,
                "positive_bank_total_inserted": positive_bank.total_inserted,
                "negative_bank_count": negative_bank.count,
                "negative_bank_total_inserted": negative_bank.total_inserted,
                "detailed_gradient_check": detailed_gradient_check,
                "elapsed_seconds_this_invocation": time.monotonic() - started,
                "learning_rate": learning_rate,
                "autocast_dtype": "bfloat16" if autocast_enabled else "float32",
                **{key: float(value.cpu()) for key, value in reduced.items()},
                **{key: float(value.cpu()) for key, value in reduced_maxima.items()},
            }
            if is_main_process(args):
                logger.log(record)
                progress.set_postfix(loss=f"{record['gmd_loss']:.4f}")

        if args.checkpoint_every and completed_step % args.checkpoint_every == 0:
            checkpoint_path = run_dir / "checkpoints" / f"checkpoint_step_{completed_step:08d}.pt"
            digest = save_checkpoint(
                checkpoint_path,
                completed_step=completed_step,
                model=train_model,
                optimizer=optimizer,
                ema=ema,
                stream=stream,
                positive_bank=positive_bank,
                negative_bank=negative_bank,
                bank_generator=bank_generator,
                prior_generator=prior_generator,
                signature=signature,
                run_dir=run_dir,
                args=args,
            )
            if is_main_process(args):
                logger.log(
                    {"event": "checkpoint", "step": completed_step, "path": str(checkpoint_path.resolve()), "sha256": digest}
                )

    final_path = run_dir / "checkpoint_final.pt"
    final_digest = save_checkpoint(
        final_path,
        completed_step=args.steps,
        model=train_model,
        optimizer=optimizer,
        ema=ema,
        stream=stream,
        positive_bank=positive_bank,
        negative_bank=negative_bank,
        bank_generator=bank_generator,
        prior_generator=prior_generator,
        signature=signature,
        run_dir=run_dir,
        args=args,
    )

    def export() -> dict[str, Any]:
        manifest = export_artifacts(
            run_dir,
            completed_step=args.steps,
            raw=unwrap_generator(train_model),
            ema=ema,
            signature=signature,
            final_checkpoint=final_path,
            final_checkpoint_sha256=final_digest,
        )
        logger.log(
            {
                "event": "complete",
                "step": args.steps,
                "elapsed_seconds_this_invocation": time.monotonic() - started,
                "final_checkpoint_sha256": final_digest,
                "strict_generator_reload": True,
                "fid_computed": False,
            }
        )
        return manifest

    run_on_rank_zero(args, export)
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    try:
        setup_distributed(args)
        return train(args)
    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
