#!/usr/bin/env python3
"""Auditable distributed MAE training for CelebA-64 pixel-space GMD.

The feature architecture lives in :mod:`src.celeba_pixel_gmd_features`.  This
file supplies the training system around it: deterministic rank-sharded image
streams, DDP, BF16 autocast, the released AdamW schedule, EMA, exact resume,
dataset/source attestation, and strict final feature artifacts.

This is a CelebA-64 adaptation of the official ImageNet-256 pixel MAE recipe,
not a numerical reproduction.  In particular, CelebA has no ImageNet class
labels, so the release's final supervised classification fine-tune is absent.
"""

from __future__ import annotations

import os
from pathlib import Path


def _configure_rank_local_miopen_cache() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cache_base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
    if cache_base is None and world_size > 1:
        token = os.environ.get("SLURM_JOB_ID", os.environ.get("MASTER_PORT", "local"))
        cache_base = f"/tmp/driftmodel_pixel_mae_miopen_{token}"
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
import hashlib
import json
import math
import platform
import random
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from PIL import Image, __version__ as PIL_VERSION
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.transforms import InterpolationMode, RandomResizedCrop
from torchvision.transforms import functional as vision_functional
from tqdm import trange

from src.celeba_pixel_gmd_features import (
    DEFAULT_EVERY_K_BLOCK,
    DEFAULT_FEATURE_PATCH_SIZES,
    DEFAULT_FEATURE_PROFILE,
    DEFAULT_INPUT_PATCH_SIZE,
    DEFAULT_LAYERS,
    DEFAULT_MASK_PATCH_SIZE,
    FEATURE_PROFILE_NONDEGENERATE_BASE,
    FEATURE_PROFILES,
    OFFICIAL_RELEASE_COMMIT,
    CelebA64PixelGMDMaskedAutoencoder,
    canonical_json_sha256,
    load_frozen_pixel_feature_encoder,
    masked_reconstruction_loss,
    sha256_file,
    state_dict_sha256,
)


REPO_ROOT = Path(__file__).resolve().parent
TRAINING_CHECKPOINT_FORMAT = "driftmodel.celeba64_pixel_gmd_mae_training"
TRAINING_CHECKPOINT_VERSION = 1
TRAINING_SIGNATURE_VERSION = 1
DATASET_MANIFEST_FORMAT = "driftmodel.celeba64_pixel_dataset_manifest"
DATASET_MANIFEST_VERSION = 1
SOURCE_SNAPSHOT_FORMAT = "driftmodel.celeba64_pixel_mae_source_snapshot"
SOURCE_SNAPSHOT_VERSION = 1
ARTIFACT_MANIFEST_FORMAT = "driftmodel.celeba64_pixel_mae_artifacts"
ARTIFACT_MANIFEST_VERSION = 1

SOURCE_FILES = (
    'celeba_pixel_gmd_mae_train.py',
    'src/celeba_pixel_gmd_features.py',
    'src/official_drifting.py',
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
AUGMENTATIONS = ("aligned_flip", "release_rrc_flip", "none")
OFFICIAL_PIXEL_MAE_CONFIG = {
    "resolution": 256,
    "base_channels": 640,
    "input_patch_size": 8,
    "mask_patch_size": 2,
    "layers": [3, 4, 6, 3],
    "global_batch": 8192,
    "updates": 200_000,
    "learning_rate": 0.004,
    "warmup_start": 1e-6,
    "warmup_updates": 4_000,
    "schedule": "constant_after_linear_warmup",
    "weight_decay": 0.01,
    "adam_betas": [0.9, 0.95],
    "mask_ratio": 0.5,
    "ema_decay": 0.9995,
    "gradient_clip": 2.0,
    "augmentation": "RandomResizedCrop(scale=(0.2,1.0)) + horizontal flip",
    "supervised_final_updates": 3_000,
    "supervised_final_lambda": 0.1,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return (
            value.detach().cpu().item()
            if value.numel() == 1
            else value.detach().cpu().tolist()
        )
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


def atomic_json_dump(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
            temporary = Path(handle.name)
            json.dump(
                _json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def atomic_text_write(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
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
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def atomic_torch_save(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _tree_hash_update(digest: Any, value: Any) -> None:
    def add(payload: bytes) -> None:
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    if torch.is_tensor(value):
        tensor = value.detach().to(device="cpu").contiguous()
        add(b"tensor")
        add(
            json.dumps(
                {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        # Reshape scalar optimizer counters before reinterpreting their bytes;
        # ``Tensor.view(dtype)`` rejects zero-dimensional tensors directly.
        add(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        return
    if isinstance(value, Mapping):
        add(b"mapping")
        encoded_keys = []
        for key in value:
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                raise TypeError("tree-hash mapping keys must be JSON scalars")
            encoded = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
            encoded_keys.append((encoded, key))
        for encoded, key in sorted(encoded_keys, key=lambda item: item[0]):
            add(encoded)
            _tree_hash_update(digest, value[key])
        return
    if isinstance(value, tuple):
        add(b"tuple")
        for item in value:
            _tree_hash_update(digest, item)
        return
    if isinstance(value, list):
        add(b"list")
        for item in value:
            _tree_hash_update(digest, item)
        return
    if isinstance(value, torch.device):
        add(b"device")
        add(str(value).encode())
        return
    if isinstance(value, torch.dtype):
        add(b"dtype")
        add(str(value).encode())
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("tree hash rejects non-finite floats")
        add(b"float")
        add(value.hex().encode())
        return
    if isinstance(value, (str, int, bool)) or value is None:
        add(type(value).__name__.encode())
        add(json.dumps(value, separators=(",", ":")).encode())
        return
    raise TypeError(f"unsupported tree-hash value: {type(value).__name__}")


def torch_tree_sha256(value: Any) -> str:
    digest = hashlib.sha256(b"driftmodel-torch-tree-v1\0")
    _tree_hash_update(digest, value)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    missing = [name for name in SOURCE_FILES if not (REPO_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required training sources are missing: {missing}")
    return {name: sha256_file(REPO_ROOT / name) for name in SOURCE_FILES}


def archive_source_snapshot(run_dir: Path, name: str) -> dict[str, Any]:
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
        relative: sha256_file(destination_root / relative)
        for relative in SOURCE_FILES
    }
    if before != after or before != copied:
        raise RuntimeError("training sources changed while snapshotting")
    manifest = {
        "format": SOURCE_SNAPSHOT_FORMAT,
        "version": SOURCE_SNAPSHOT_VERSION,
        "created_at_utc": utc_now(),
        "directory": str(destination_root.resolve()),
        "sha256": copied,
    }
    manifest_path = atomic_json_dump(destination_root / "manifest.json", manifest)
    return {
        **manifest,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _git_provenance() -> dict[str, Any]:
    def run(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *arguments],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "status_porcelain": run("status", "--short"),
    }


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: Mapping[str, Any]) -> None:
        payload = {"timestamp_utc": utc_now(), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    _json_safe(payload), sort_keys=True, allow_nan=False
                )
                + "\n"
            )
            handle.flush()


def _parse_layers(raw: str) -> tuple[int, int, int, int]:
    try:
        parsed = tuple(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("layers must be four comma-separated integers") from error
    if len(parsed) != 4 or any(value < 1 for value in parsed):
        raise argparse.ArgumentTypeError("layers must be four positive integers")
    return parsed  # type: ignore[return-value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Auditable DDP CelebA-64 pixel MAE for GMD features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "celeba64" / "celeba64_train",
    )
    parser.add_argument("--train-start-index", type=int, default=5_000)
    parser.add_argument("--train-end-index", type=int, default=30_000)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument(
        "--global-batch",
        type=int,
        default=256,
        help="batch over all torchrun ranks",
    )
    parser.add_argument(
        "--preload-uint8",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="decode the selected 64px dataset once per rank into CPU uint8",
    )
    parser.add_argument(
        "--augmentation",
        choices=AUGMENTATIONS,
        default="aligned_flip",
        help=(
            "aligned_flip preserves registered faces; release_rrc_flip applies the "
            "official crop operator to already-aligned 64px files"
        ),
    )

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--layers", type=_parse_layers, default=DEFAULT_LAYERS)
    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument("--gn-max-groups", type=int, default=32)
    parser.add_argument(
        "--feature-profile", choices=FEATURE_PROFILES, default=DEFAULT_FEATURE_PROFILE
    )
    parser.add_argument("--mask-ratio-min", type=float, default=0.5)
    parser.add_argument("--mask-ratio-max", type=float, default=0.5)

    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--lr-warmup-start", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=4_000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument(
        "--use-bfloat16", action=argparse.BooleanOptionalAction, default=True
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--distributed-timeout-minutes", type=int, default=30)

    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "celeba_pixel_gmd" / "features",
    )
    parser.add_argument("--run-label", default="celeba64_pixel_mae")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--resume-sha256",
        default=None,
        help="required external SHA-256 of --resume; the sidecar is checked too",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument(
        "--verify-gradients",
        action="store_true",
        help="require finite complete gradients and synchronized DDP fingerprints",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU/Gloo configuration, including strict checkpoints",
    )
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
        args.train_start_index = 0
        args.train_end_index = min(args.train_end_index or 16, 16)
        args.base_channels = min(args.base_channels, 4)
        args.layers = (1, 1, 1, 1)
        args.gn_max_groups = min(args.gn_max_groups, 4)
        args.feature_profile = FEATURE_PROFILE_NONDEGENERATE_BASE
        args.use_bfloat16 = False
        args.preload_uint8 = False
        args.augmentation = "aligned_flip"
        args.lr_warmup_steps = 0
        args.log_every = 1
        args.checkpoint_every = 1
        args.verify_gradients = True
    return args


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.is_dir() or args.data_dir.is_symlink():
        raise FileNotFoundError("--data-dir must be a regular directory, not a symlink")
    if args.steps < 1 or args.global_batch < 1:
        raise ValueError("steps and global batch must be positive")
    if not args.smoke and not args.preload_uint8:
        raise ValueError(
            "audited non-smoke training requires --preload-uint8 so dataset bytes "
            "cannot change underneath an active run"
        )
    if args.global_batch % args.world_size:
        raise ValueError("global batch must be divisible by world size")
    args.local_batch = args.global_batch // args.world_size
    if args.local_batch < 1:
        raise ValueError("local batch must be positive")
    if args.train_start_index < 0:
        raise ValueError("train start index must be non-negative")
    if args.train_end_index is not None and args.train_end_index <= args.train_start_index:
        raise ValueError("train end index must exceed train start index")
    if args.base_channels < 4:
        raise ValueError("base channels must be at least four")
    if args.gn_max_groups < 1:
        raise ValueError("GroupNorm maximum must be positive")
    if not 0 <= args.dropout_prob < 1:
        raise ValueError("dropout probability must lie in [0,1)")
    if not 0 <= args.mask_ratio_min <= args.mask_ratio_max <= 1:
        raise ValueError("mask ratios must satisfy 0 <= min <= max <= 1")
    if args.mask_ratio_max == 0:
        raise ValueError("a zero mask makes masked-only reconstruction identically zero")
    positive_names = ("lr", "adam_eps", "grad_clip")
    for name in positive_names:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
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
    else:
        if not args.resume.is_file() or args.resume.is_symlink():
            raise FileNotFoundError("--resume must be a regular checkpoint file")
        if not _valid_sha256(args.resume_sha256):
            raise ValueError("--resume requires a lowercase 64-character --resume-sha256")


def setup_distributed(args: argparse.Namespace) -> None:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1
    if args.smoke:
        args.global_batch = min(args.global_batch, 2 * args.world_size)
        torch.set_num_threads(1)
    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested == "auto":
        requested = "cpu"
    if args.distributed:
        if requested == "cuda":
            if args.distributed_backend != "nccl":
                raise ValueError("GPU DDP requires NCCL/RCCL")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/ROCm requested but unavailable")
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        elif requested == "cpu":
            if args.distributed_backend != "gloo":
                raise ValueError("CPU DDP requires Gloo")
            args.device = torch.device("cpu")
        else:
            raise ValueError(f"unsupported distributed device: {requested}")
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


def seed_initial_parameters(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def seed_rank_stochastic_stream(args: argparse.Namespace) -> None:
    random.seed(args.seed + args.rank)
    np.random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed + args.rank)


def _discover_selected_paths(
    data_dir: Path,
    start_index: int,
    end_index: int | None,
) -> list[Path]:
    paths = sorted(
        path
        for path in data_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no CelebA images in {data_dir}")
    if any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("dataset images must be regular files, not symlinks")
    stop = len(paths) if end_index is None else end_index
    if stop > len(paths):
        raise ValueError(
            f"train end index {stop} exceeds the {len(paths)} discovered images"
        )
    selected = paths[start_index:stop]
    if not selected:
        raise ValueError("selected CelebA training split is empty")
    return selected


def build_dataset_manifest(
    data_dir: Path,
    start_index: int,
    end_index: int | None,
) -> dict[str, Any]:
    selected = _discover_selected_paths(data_dir, start_index, end_index)
    records = []
    total_bytes = 0
    for path in selected:
        size = path.stat().st_size
        total_bytes += size
        records.append(
            {
                "relative_path": path.relative_to(data_dir).as_posix(),
                "bytes": size,
                "sha256": sha256_file(path),
            }
        )
    files_sha256 = canonical_json_sha256(records)
    return {
        "format": DATASET_MANIFEST_FORMAT,
        "version": DATASET_MANIFEST_VERSION,
        "root": str(data_dir.resolve()),
        "selection_semantics": "sorted image filenames, Python half-open slice",
        "train_start_index": start_index,
        "train_end_index": start_index + len(selected),
        "image_count": len(selected),
        "total_bytes": total_bytes,
        "image_shape": [3, 64, 64],
        "source_encoding": "RGB uint8 PNG/JPEG decoded by Pillow",
        "files_sha256": files_sha256,
        "files": records,
    }


def _load_rgb_uint8(path: Path) -> Tensor:
    with Image.open(path) as image:
        converted = image.convert("RGB")
        if converted.size != (64, 64):
            raise ValueError(f"CelebA-64 file has size {converted.size}: {path}")
        array = np.array(converted, dtype=np.uint8, copy=True)
    if array.shape != (64, 64, 3):
        raise ValueError(f"unexpected decoded RGB shape {array.shape}: {path}")
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CelebA64PixelDataset:
    def __init__(
        self,
        data_dir: Path,
        manifest: Mapping[str, Any],
        *,
        preload_uint8: bool,
    ) -> None:
        records = manifest.get("files")
        if not isinstance(records, list) or not records:
            raise ValueError("dataset manifest lacks files")
        self.paths = [data_dir / str(record["relative_path"]) for record in records]
        self.preloaded: Tensor | None = None
        if preload_uint8:
            self.preloaded = torch.stack([_load_rgb_uint8(path) for path in self.paths])
            if self.preloaded.dtype != torch.uint8:
                raise RuntimeError("preloaded dataset must remain uint8")

    def __len__(self) -> int:
        return len(self.paths)

    def batch(self, indices: Tensor) -> Tensor:
        if indices.device.type != "cpu" or indices.dtype != torch.int64:
            raise ValueError("dataset indices must be CPU int64")
        if self.preloaded is not None:
            return self.preloaded.index_select(0, indices)
        return torch.stack([_load_rgb_uint8(self.paths[index]) for index in indices.tolist()])


class PixelBatchStream:
    """Global epoch permutations with disjoint rank slices and exact resume."""

    def __init__(self, dataset: CelebA64PixelDataset, args: argparse.Namespace) -> None:
        self.dataset = dataset
        self.seed = int(args.seed)
        self.rank = int(args.rank)
        self.world_size = int(args.world_size)
        self.global_batch = int(args.global_batch)
        self.local_batch = int(args.local_batch)
        self.batches_per_epoch = len(dataset) // self.global_batch
        if self.batches_per_epoch < 1:
            raise ValueError("training split is smaller than one global batch")
        self.usable_count = self.batches_per_epoch * self.global_batch
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
        start = self.batch_offset * self.global_batch + self.rank * self.local_batch
        indices = self._permutation[start : start + self.local_batch]
        self.batch_offset += 1
        return indices

    def next(self) -> Tensor:
        return self.dataset.batch(self.next_indices())

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_offset": self.batch_offset}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"epoch", "batch_offset"}:
            raise ValueError("malformed pixel-stream state")
        epoch = int(state["epoch"])
        batch_offset = int(state["batch_offset"])
        if epoch < 0 or not 0 <= batch_offset <= self.batches_per_epoch:
            raise ValueError("invalid pixel-stream position")
        self.epoch = epoch
        self.batch_offset = batch_offset
        self._permutation = self._make_permutation()


def apply_augmentation(images_uint8: Tensor, mode: str) -> Tensor:
    """Return CPU FP32 RGB in [-1,1], consuming only the CPU torch RNG."""

    if images_uint8.device.type != "cpu" or images_uint8.dtype != torch.uint8:
        raise ValueError("augmentation expects CPU uint8 images")
    if mode not in AUGMENTATIONS:
        raise ValueError(f"unsupported augmentation {mode!r}")
    images = images_uint8.float().div_(255.0)
    if mode == "release_rrc_flip":
        cropped = []
        for image in images:
            top, left, height, width = RandomResizedCrop.get_params(
                image,
                scale=(0.2, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            )
            cropped.append(
                vision_functional.resized_crop(
                    image,
                    top,
                    left,
                    height,
                    width,
                    [64, 64],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
            )
        images = torch.stack(cropped)
        # Tensor bicubic interpolation may overshoot slightly, whereas the
        # release resizes PIL uint8 images before ToTensor and is therefore
        # saturated to the representable image interval.
        images.clamp_(0.0, 1.0)
    if mode in {"aligned_flip", "release_rrc_flip"}:
        flip = torch.rand(images.shape[0]) < 0.5
        images = torch.where(flip[:, None, None, None], images.flip(-1), images)
    return images.mul_(2.0).sub_(1.0)


def make_model(args: argparse.Namespace) -> CelebA64PixelGMDMaskedAutoencoder:
    return CelebA64PixelGMDMaskedAutoencoder(
        base_channels=args.base_channels,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        gn_max_groups=args.gn_max_groups,
        input_patch_size=DEFAULT_INPUT_PATCH_SIZE,
        mask_patch_size=DEFAULT_MASK_PATCH_SIZE,
        feature_patch_sizes=DEFAULT_FEATURE_PATCH_SIZES,
        every_k_block=DEFAULT_EVERY_K_BLOCK,
        feature_profile=args.feature_profile,
    )


def unwrap_model(model: nn.Module) -> CelebA64PixelGMDMaskedAutoencoder:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(unwrapped, CelebA64PixelGMDMaskedAutoencoder):
        raise TypeError("unexpected pixel MAE model")
    return unwrapped


def make_ema(model: CelebA64PixelGMDMaskedAutoencoder) -> CelebA64PixelGMDMaskedAutoencoder:
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(
    ema: CelebA64PixelGMDMaskedAutoencoder,
    model: CelebA64PixelGMDMaskedAutoencoder,
    decay: float,
) -> None:
    for ema_parameter, parameter in zip(
        ema.parameters(), model.parameters(), strict=True
    ):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema.buffers(), model.buffers(), strict=True):
        ema_buffer.copy_(buffer)


def learning_rate_at_step(step_index: int, args: argparse.Namespace) -> float:
    """Match Optax linear warmup at counts [0,warmup), then constant LR."""

    if args.lr_warmup_steps and step_index < args.lr_warmup_steps:
        progress = step_index / args.lr_warmup_steps
        return args.lr_warmup_start + progress * (args.lr - args.lr_warmup_start)
    return args.lr


def local_rng_state(args: argparse.Namespace) -> dict[str, Any]:
    # All stochastic training operations (augmentation, mask ratio and mask)
    # use torch. Pillow decoding and NumPy conversion are deterministic and do
    # not consume their module RNGs, so only the causally relevant streams are
    # checkpointed. This also keeps the payload compatible with
    # ``torch.load(weights_only=True)``.
    state: dict[str, Any] = {"cpu": torch.get_rng_state().cpu()}
    if args.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(args.device).cpu()
    return state


def gather_rank_objects(local: Any, args: argparse.Namespace) -> list[Any]:
    if not args.distributed:
        return [local]
    gathered: list[Any] = [None] * args.world_size
    dist.all_gather_object(gathered, local)
    if any(value is None for value in gathered):
        raise RuntimeError("failed to gather every rank object")
    return gathered


def restore_rng_state(states: Any, args: argparse.Namespace) -> None:
    if not isinstance(states, list) or len(states) != args.world_size:
        raise ValueError("checkpoint RNG world size differs from launch")
    state = states[args.rank]
    if not isinstance(state, Mapping) or set(state) not in (
        {"cpu"},
        {"cpu", "cuda"},
    ):
        raise ValueError("checkpoint has malformed rank RNG state")
    if not torch.is_tensor(state["cpu"]):
        raise ValueError("checkpoint CPU RNG state is not a tensor")
    torch.set_rng_state(state["cpu"].cpu())
    if args.device.type == "cuda":
        if not torch.is_tensor(state.get("cuda")):
            raise ValueError("checkpoint lacks CUDA/ROCm RNG state")
        torch.cuda.set_rng_state(state["cuda"].cpu(), args.device)


def reduce_stats(stats: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Tensor]:
    keys = list(stats)
    packed = torch.stack(
        [
            (
                value.detach().to(device=args.device, dtype=torch.float32).reshape(())
                if torch.is_tensor(value)
                else torch.tensor(float(value), device=args.device)
            )
            for value in stats.values()
        ]
    )
    if args.distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(args.world_size)
    return {key: packed[index] for index, key in enumerate(keys)}


def verify_gradients(model: nn.Module, args: argparse.Namespace) -> None:
    if not args.verify_gradients:
        return
    gradients = [parameter.grad for parameter in model.parameters()]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("at least one trainable MAE parameter has no gradient")
    assert all(gradient is not None for gradient in gradients)
    if not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
        raise FloatingPointError("non-finite MAE gradient")
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
            raise RuntimeError("DDP gradient fingerprints differ across ranks")


def training_signature(
    args: argparse.Namespace,
    dataset_manifest: Mapping[str, Any],
    model: CelebA64PixelGMDMaskedAutoencoder,
) -> dict[str, Any]:
    return {
        "version": TRAINING_SIGNATURE_VERSION,
        "objective": "masked_only_packed_pixel_reconstruction",
        "architecture": model.architecture_config(),
        "feature_schema": model.feature_schema(),
        "input_model_space": "RGB [-1,1]",
        "mask_sampling": "independent Bernoulli on 2x2 packed-grid patches",
        "mask_ratio_min": args.mask_ratio_min,
        "mask_ratio_max": args.mask_ratio_max,
        "augmentation": args.augmentation,
        "augmentation_source_notice": (
            "release_rrc_flip operates on already aligned 64px files, not original ImageNet-like sources"
        ),
        "classification_head": "absent_unconditional_celeba_adaptation",
        "optimizer": "torch.optim.AdamW",
        "learning_rate": args.lr,
        "learning_rate_schedule": "linear_1e-6_style_warmup_then_constant",
        "lr_warmup_start": args.lr_warmup_start,
        "lr_warmup_steps": args.lr_warmup_steps,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "ema_initialized_from": "theta_0",
        "use_bfloat16": args.use_bfloat16,
        "bf16_semantics": "CUDA/ROCm autocast; FP32 parameters, optimizer, loss accumulation",
        "global_batch": args.global_batch,
        "local_batch": args.local_batch,
        "world_size": args.world_size,
        "ddp_broadcast_buffers": False,
        "device_type": args.device.type,
        "accelerator_name": (
            torch.cuda.get_device_name(args.device)
            if args.device.type == "cuda"
            else "cpu"
        ),
        "distributed_backend": args.distributed_backend,
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision.__version__),
        "pillow_version": str(PIL_VERSION),
        "numpy_version": str(np.__version__),
        "seed": args.seed,
        "deterministic_algorithms": args.deterministic,
        "preload_uint8": args.preload_uint8,
        "dataset_root": dataset_manifest["root"],
        "dataset_files_sha256": dataset_manifest["files_sha256"],
        "dataset_image_count": dataset_manifest["image_count"],
        "train_start_index": dataset_manifest["train_start_index"],
        "train_end_index": dataset_manifest["train_end_index"],
        "source_files_sha256": _source_hashes(),
        "official_release_commit": OFFICIAL_RELEASE_COMMIT,
    }


def resolve_run_directory(args: argparse.Namespace) -> Path:
    def resolve() -> str:
        if args.resume is not None:
            checkpoint_parent = args.resume.resolve().parent
            inferred = (
                checkpoint_parent.parent
                if checkpoint_parent.name == "checkpoints"
                else checkpoint_parent
            )
            if args.run_dir is not None and args.run_dir.resolve() != inferred:
                raise ValueError("--run-dir differs from the directory implied by --resume")
            if not inferred.is_dir():
                raise FileNotFoundError(f"resume run directory is absent: {inferred}")
            path = inferred
        elif args.run_dir is not None:
            path = args.run_dir.resolve()
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"new run directory is not empty: {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            label = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in args.run_label
            ).strip("_")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = (args.output_root / f"{stamp}_{label}").resolve()
            path.mkdir(parents=True, exist_ok=False)
        return str(path)

    return Path(run_on_rank_zero(args, resolve))


def _checkpoint_integrity(
    *,
    model_state: Mapping[str, Tensor],
    ema_state: Mapping[str, Tensor],
    optimizer_state: Mapping[str, Any],
    stream_state: Mapping[str, Any],
    rng_states: Sequence[Any],
    signature: Mapping[str, Any],
) -> dict[str, str]:
    return {
        "algorithm": "sha256",
        "model_state_sha256": state_dict_sha256(model_state),
        "ema_state_sha256": state_dict_sha256(ema_state),
        "optimizer_state_sha256": torch_tree_sha256(optimizer_state),
        "stream_state_sha256": torch_tree_sha256(stream_state),
        "rng_states_sha256": torch_tree_sha256(list(rng_states)),
        "training_signature_sha256": canonical_json_sha256(dict(signature)),
    }


def save_checkpoint(
    path: Path,
    *,
    completed_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: CelebA64PixelGMDMaskedAutoencoder,
    stream: PixelBatchStream,
    signature: Mapping[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> str:
    rng_states = gather_rank_objects(local_rng_state(args), args)
    stream_states = gather_rank_objects(stream.state_dict(), args)
    if any(state != stream_states[0] for state in stream_states[1:]):
        raise RuntimeError("rank data-stream positions diverged")

    def save() -> str:
        model_state = unwrap_model(model).state_dict()
        ema_state = ema.state_dict()
        optimizer_state = optimizer.state_dict()
        stream_state = stream_states[0]
        payload = {
            "format": TRAINING_CHECKPOINT_FORMAT,
            "version": TRAINING_CHECKPOINT_VERSION,
            "completed_step": completed_step,
            "model": model_state,
            "ema": ema_state,
            "optimizer": optimizer_state,
            "data_stream": stream_state,
            "rng_by_rank": rng_states,
            "training_signature": dict(signature),
            "run_dir": str(run_dir.resolve()),
            "saved_at_utc": utc_now(),
            "integrity": _checkpoint_integrity(
                model_state=model_state,
                ema_state=ema_state,
                optimizer_state=optimizer_state,
                stream_state=stream_state,
                rng_states=rng_states,
                signature=signature,
            ),
        }
        atomic_torch_save(path, payload)
        digest = sha256_file(path)
        atomic_text_write(
            Path(f"{path}.sha256"), f"{digest}  {path.name}\n"
        )
        return digest

    return str(run_on_rank_zero(args, save))


def _verified_checkpoint_file(path: Path, expected_sha256: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("resume checkpoint must be a regular file")
    sidecar = Path(f"{path}.sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise FileNotFoundError(f"resume checkpoint sidecar is absent: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name or not _valid_sha256(fields[0]):
        raise ValueError("resume checkpoint sidecar is malformed")
    observed = sha256_file(path)
    if observed != fields[0] or observed != expected_sha256:
        raise ValueError("resume checkpoint file SHA-256 mismatch")
    return observed


def _validate_checkpoint_integrity(checkpoint: Mapping[str, Any]) -> None:
    expected_keys = {
        "algorithm",
        "model_state_sha256",
        "ema_state_sha256",
        "optimizer_state_sha256",
        "stream_state_sha256",
        "rng_states_sha256",
        "training_signature_sha256",
    }
    integrity = checkpoint.get("integrity")
    if not isinstance(integrity, Mapping) or set(integrity) != expected_keys:
        raise ValueError("resume checkpoint integrity schema differs")
    if integrity["algorithm"] != "sha256":
        raise ValueError("resume checkpoint integrity algorithm differs")
    calculated = _checkpoint_integrity(
        model_state=checkpoint["model"],
        ema_state=checkpoint["ema"],
        optimizer_state=checkpoint["optimizer"],
        stream_state=checkpoint["data_stream"],
        rng_states=checkpoint["rng_by_rank"],
        signature=checkpoint["training_signature"],
    )
    if dict(integrity) != calculated:
        differing = sorted(
            key for key in expected_keys if integrity.get(key) != calculated.get(key)
        )
        raise ValueError(f"resume checkpoint internal SHA-256 differs for {differing}")


def load_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: CelebA64PixelGMDMaskedAutoencoder,
    stream: PixelBatchStream,
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
        "rng_by_rank",
        "training_signature",
        "run_dir",
        "saved_at_utc",
        "integrity",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != expected_top:
        raise ValueError("--resume expects the exact pixel-MAE checkpoint schema")
    if checkpoint["format"] != TRAINING_CHECKPOINT_FORMAT:
        raise ValueError("--resume is not a CelebA pixel-MAE checkpoint")
    if checkpoint["version"] != TRAINING_CHECKPOINT_VERSION:
        raise ValueError("unsupported pixel-MAE checkpoint version")
    _validate_checkpoint_integrity(checkpoint)
    if checkpoint["training_signature"] != dict(signature):
        found = checkpoint["training_signature"]
        found = found if isinstance(found, Mapping) else {}
        differing = sorted(
            key
            for key in set(found).union(signature)
            if found.get(key) != signature.get(key)
        )
        raise ValueError(f"resume training signature differs for {differing}")
    if Path(str(checkpoint["run_dir"])).resolve() != run_dir.resolve():
        raise ValueError("resume run-directory provenance differs")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    ema.load_state_dict(checkpoint["ema"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    stream.load_state_dict(checkpoint["data_stream"])
    restore_rng_state(checkpoint["rng_by_rank"], args)
    completed_step = int(checkpoint["completed_step"])
    if completed_step < 0:
        raise ValueError("resume checkpoint has a negative step")
    return completed_step


def write_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    dataset_manifest: Mapping[str, Any],
    signature: Mapping[str, Any],
    model: CelebA64PixelGMDMaskedAutoencoder,
) -> None:
    resume_token = (
        ""
        if args.resume is None
        else f"_resume_{utc_now().replace(':', '-').replace('+', '_')}"
    )
    snapshot = archive_source_snapshot(run_dir, f"source_snapshot{resume_token}")
    dataset_path = run_dir / f"dataset_manifest{resume_token}.json"
    atomic_json_dump(dataset_path, dataset_manifest)
    config = {
        **{
            key: value
            for key, value in vars(args).items()
            if key not in {"resume_sha256"} and not key.startswith("_")
        },
        "resume_sha256": args.resume_sha256,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": model.architecture_config(),
        "feature_schema": model.feature_schema(),
        "training_signature": signature,
        "requested_total_steps_not_in_resume_signature": args.steps,
    }
    atomic_json_dump(run_dir / f"config{resume_token}.json", config)
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
        "device": str(args.device),
        "world_size": args.world_size,
        "global_batch": args.global_batch,
        "git": _git_provenance(),
        "source_snapshot": snapshot,
        "dataset_manifest": {
            "path": str(dataset_path.resolve()),
            "sha256": sha256_file(dataset_path),
            "files_sha256": dataset_manifest["files_sha256"],
            "image_count": dataset_manifest["image_count"],
        },
        "official_release_commit": OFFICIAL_RELEASE_COMMIT,
        "official_pixel_mae_config": OFFICIAL_PIXEL_MAE_CONFIG,
        "adaptations": {
            "dataset": "CelebA aligned 64px instead of ImageNet-256",
            "packing": "2x2 RGB to 12x32x32 instead of 8x8 RGB to 192x32x32",
            "classes": "unconditional; classification head and final supervised fine-tune absent",
            "width": f"base_channels={args.base_channels}, official is 640",
            "global_batch": f"{args.global_batch}, official is 8192",
            "augmentation": args.augmentation,
            "feature_profile": args.feature_profile,
            "framework": "PyTorch DDP/AdamW instead of JAX/Flax/Optax",
        },
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_NODELIST",
                "SLURM_PROCID",
                "SLURM_LOCALID",
            )
        },
    }
    atomic_json_dump(run_dir / f"provenance{resume_token}.json", provenance)


def _export_feature_artifacts(
    run_dir: Path,
    *,
    completed_step: int,
    raw: CelebA64PixelGMDMaskedAutoencoder,
    ema: CelebA64PixelGMDMaskedAutoencoder,
    signature: Mapping[str, Any],
    final_training_checkpoint: Path,
    final_training_checkpoint_sha256: str,
) -> dict[str, Any]:
    common = {
        "completed_step": completed_step,
        "training_signature": dict(signature),
        "training_signature_sha256": canonical_json_sha256(dict(signature)),
        "trained_at_utc": utc_now(),
    }
    raw_path = run_dir / "feature_encoder_raw_final.pt"
    ema_path = run_dir / "feature_encoder_ema_final.pt"
    atomic_torch_save(
        raw_path,
        raw.checkpoint_dict(metadata={**common, "weight_kind": "raw"}),
    )
    atomic_torch_save(
        ema_path,
        ema.checkpoint_dict(
            metadata={
                **common,
                "weight_kind": "ema",
                "ema_initialized_from": "theta_0",
                "ema_decay": signature["ema_decay"],
            }
        ),
    )
    # Re-read both payloads through the strict phase-1 loader before attesting.
    load_frozen_pixel_feature_encoder(raw_path, "cpu")
    load_frozen_pixel_feature_encoder(ema_path, "cpu")
    artifacts = {
        "training_checkpoint": {
            "path": str(final_training_checkpoint.resolve()),
            "sha256": final_training_checkpoint_sha256,
        },
        "raw_feature_encoder": {
            "path": str(raw_path.resolve()),
            "sha256": sha256_file(raw_path),
        },
        "ema_feature_encoder": {
            "path": str(ema_path.resolve()),
            "sha256": sha256_file(ema_path),
        },
    }
    manifest = {
        "format": ARTIFACT_MANIFEST_FORMAT,
        "version": ARTIFACT_MANIFEST_VERSION,
        "created_at_utc": utc_now(),
        "completed_step": completed_step,
        "training_signature_sha256": canonical_json_sha256(dict(signature)),
        "strict_feature_reload": True,
        "artifacts": artifacts,
    }
    path = atomic_json_dump(run_dir / "artifact_manifest.json", manifest)
    atomic_text_write(
        run_dir / "artifact_manifest.json.sha256",
        f"{sha256_file(path)}  {path.name}\n",
    )
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
        args.data_dir.resolve(),
        dataset_manifest,
        preload_uint8=args.preload_uint8,
    )
    # Every rank has finished decoding before rank zero re-hashes the source
    # tree. Non-smoke runs train only from the now-frozen uint8 tensor, so a
    # matching second manifest closes the hash/decode time-of-check window.
    verified_manifest = run_on_rank_zero(
        args,
        lambda: build_dataset_manifest(
            args.data_dir.resolve(), args.train_start_index, args.train_end_index
        ),
    )
    if verified_manifest != dataset_manifest:
        raise RuntimeError("CelebA dataset changed while it was being preloaded")
    stream = PixelBatchStream(dataset, args)

    seed_initial_parameters(args)
    model = make_model(args).to(args.device)
    train_model: nn.Module = model
    if args.distributed:
        ddp_options: dict[str, Any] = {"broadcast_buffers": False}
        if args.device.type == "cuda":
            ddp_options.update(
                device_ids=[args.local_rank], output_device=args.local_rank
            )
        train_model = DistributedDataParallel(model, **ddp_options)
    seed_rank_stochastic_stream(args)
    optimizer = torch.optim.AdamW(
        train_model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    ema = make_ema(unwrap_model(train_model))
    signature = training_signature(args, dataset_manifest, unwrap_model(train_model))
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
            signature=signature,
            run_dir=run_dir,
            args=args,
        )
    if start_step >= args.steps:
        raise ValueError(
            f"checkpoint is at step {start_step}; --steps must be larger"
        )

    run_on_rank_zero(
        args,
        lambda: write_metadata(
            args,
            run_dir,
            dataset_manifest,
            signature,
            unwrap_model(train_model),
        ),
    )
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    if is_main_process(args):
        logger.log(
            {
                "event": "resume" if start_step else "start",
                "step": start_step,
                "run_dir": str(run_dir),
                "dataset_files_sha256": dataset_manifest["files_sha256"],
                "train_count": len(dataset),
                "global_batch": args.global_batch,
                "local_batch": args.local_batch,
                "world_size": args.world_size,
                "augmentation": args.augmentation,
                "feature_profile": args.feature_profile,
                "model_parameters": sum(
                    parameter.numel()
                    for parameter in unwrap_model(train_model).parameters()
                ),
            }
        )

    started = time.monotonic()
    last_reduced: dict[str, Tensor] = {}
    progress = trange(
        start_step,
        args.steps,
        initial=start_step,
        total=args.steps,
        desc="celeba64-pixel:gmd-mae",
        disable=not is_main_process(args),
    )
    for step_index in progress:
        learning_rate = learning_rate_at_step(step_index, args)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        images_cpu = apply_augmentation(stream.next(), args.augmentation)
        images = images_cpu.to(
            args.device,
            non_blocking=args.device.type == "cuda",
        )
        ratio = torch.rand(
            images.shape[0], device=args.device, dtype=torch.float32
        ) * (args.mask_ratio_max - args.mask_ratio_min) + args.mask_ratio_min
        mask = unwrap_model(train_model).sample_mask(images, ratio)

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = args.use_bfloat16 and args.device.type == "cuda"
        with torch.autocast(
            device_type=args.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            prediction = train_model(images, mask)
            per_sample_loss = masked_reconstruction_loss(
                prediction, images, mask
            )
            loss = per_sample_loss.mean()
        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        if args.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"non-finite MAE loss at step {step_index + 1}")
        loss.backward()
        verify_gradients(train_model, args)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            train_model.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        update_ema(ema, unwrap_model(train_model), args.ema_decay)
        completed_step = step_index + 1

        should_log = (
            completed_step == 1
            or completed_step == args.steps
            or completed_step % args.log_every == 0
        )
        if should_log:
            last_reduced = reduce_stats(
                {
                    "loss": loss,
                    "loss_sample_std": per_sample_loss.detach().float().std(
                        correction=0
                    ),
                    "mask_ratio": mask.float().mean(),
                    "input_mean": images.float().mean(),
                    "input_std": images.float().std(correction=0),
                    "prediction_mean": prediction.detach().float().mean(),
                    "prediction_std": prediction.detach().float().std(correction=0),
                    "grad_norm_preclip": grad_norm,
                },
                args,
            )
            record = {
                "event": "train",
                "step": completed_step,
                "epoch": stream.epoch,
                "batch_offset": stream.batch_offset,
                "examples_seen_this_invocation": (
                    completed_step - start_step
                )
                * args.global_batch,
                "elapsed_seconds_this_invocation": time.monotonic() - started,
                "learning_rate": learning_rate,
                "autocast_dtype": "bfloat16" if autocast_enabled else "float32",
                **{
                    key: float(value.detach().cpu())
                    for key, value in last_reduced.items()
                },
            }
            if is_main_process(args):
                logger.log(record)
                progress.set_postfix(loss=f"{record['loss']:.4f}")

        if args.checkpoint_every and completed_step % args.checkpoint_every == 0:
            checkpoint_path = (
                run_dir
                / "checkpoints"
                / f"checkpoint_step_{completed_step:08d}.pt"
            )
            checkpoint_sha = save_checkpoint(
                checkpoint_path,
                completed_step=completed_step,
                model=train_model,
                optimizer=optimizer,
                ema=ema,
                stream=stream,
                signature=signature,
                run_dir=run_dir,
                args=args,
            )
            if is_main_process(args):
                logger.log(
                    {
                        "event": "checkpoint",
                        "step": completed_step,
                        "path": str(checkpoint_path.resolve()),
                        "sha256": checkpoint_sha,
                    }
                )

    final_path = run_dir / "checkpoint_final.pt"
    final_sha = save_checkpoint(
        final_path,
        completed_step=args.steps,
        model=train_model,
        optimizer=optimizer,
        ema=ema,
        stream=stream,
        signature=signature,
        run_dir=run_dir,
        args=args,
    )

    def export() -> dict[str, Any]:
        manifest = _export_feature_artifacts(
            run_dir,
            completed_step=args.steps,
            raw=unwrap_model(train_model),
            ema=ema,
            signature=signature,
            final_training_checkpoint=final_path,
            final_training_checkpoint_sha256=final_sha,
        )
        logger.log(
            {
                "event": "complete",
                "step": args.steps,
                "elapsed_seconds_this_invocation": time.monotonic() - started,
                "final_checkpoint_sha256": final_sha,
                "strict_feature_reload": True,
                "artifact_manifest": str(
                    (run_dir / "artifact_manifest.json").resolve()
                ),
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
