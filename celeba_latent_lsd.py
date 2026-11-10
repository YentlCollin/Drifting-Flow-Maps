#!/usr/bin/env python3
"""Train a CelebA latent flow map with Lagrangian self-distillation.

This is an isolated latent-space adaptation of Boffi, Albergo and
Vanden-Eijnden's released LSD objective.  The baseline uses independent
``x0 ~ N(0,I)`` and true cached CelebA latents, an exact 75/25 local split,
the current raw parameters as the stopped target, and
``X(x,s,t)=x+(t-s)phi(x,s,t)``.  No WGF, semigroup, GMD, endpoint-teacher, or
EMA-teacher loss is mixed into that baseline.

The optional ``teacher_coupled_ablation`` changes the interpolation coupling
itself to ``x1=stopgrad(T_GMD(x0,0,1))`` using the same ``x0``.  It is not the
paper's data-paired LSD experiment and is labelled as an ablation everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil


def _configure_rank_local_miopen_cache() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cache_base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
    if cache_base is None and world_size > 1:
        token = os.environ.get("SLURM_JOB_ID", os.environ.get("MASTER_PORT", "local"))
        cache_base = f"/tmp/driftmodel_lsd_miopen_{token}"
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
import hashlib
import json
import math
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from tqdm import trange

from local_paths import REPO_ROOT
from src.celeba_latent import (
    CELEBA_LATENT_DIM,
    TransitionConditionedLatentUNet,
    flatten_latents,
    load_latent_cache,
    sha256_tensor,
)
from src.celeba_latent_lsd import (
    CelebALatentLSDBackbone,
    CelebALatentLSDTrainingModel,
    project_log_variance_weight_,
)
from src.flow_map_self_distillation import (
    MagnitudePreservingTimeLogVariance,
    ResidualTwoTimeFlowMap,
)


CHECKPOINT_FORMAT = "driftmodel.celeba_latent_lsd"
CHECKPOINT_VERSION = 1
WEIGHTS_FORMAT = "driftmodel.celeba_latent_lsd_weights"
WEIGHTS_VERSION = 1
SIGNATURE_VERSION = 1
DIAGONAL_FRACTION = 0.75
OFFICIAL_LSD_COMMIT = "2f115a07fa9073553193e4b265dfc303827af2b0"
SOURCE_FILES = (
    'celeba_latent_lsd.py',
    'src/celeba_latent_lsd.py',
    'src/flow_map_self_distillation.py',
    'src/celeba_latent.py',
    'local_paths.py',
    'pyproject.toml',
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def atomic_json_dump(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def atomic_torch_save(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
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
    for relative_path in SOURCE_FILES:
        destination = destination_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative_path, destination)
    after = _source_hashes()
    copied = {
        name: sha256_file(destination_root / name)
        for name in SOURCE_FILES
    }
    if before != after or before != copied:
        raise RuntimeError("training sources changed while the snapshot was archived")
    manifest = {
        "created_at_utc": utc_now(),
        "directory": str(destination_root),
        "sha256": copied,
    }
    atomic_json_dump(destination_root / "manifest.json", manifest)
    return manifest


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
            handle.write(json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CelebA latent flow-map self-distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "data" / "celeba64_latents_sdvae_ft_mse.pt",
    )
    parser.add_argument("--train-start-index", type=int, default=5_000)
    parser.add_argument("--train-end-index", type=int, default=30_000)
    parser.add_argument("--steps", type=int, default=800_000)
    parser.add_argument(
        "--global-batch",
        type=int,
        default=256,
        help="global batch across all torchrun ranks; must be divisible by 4 and world size",
    )
    parser.add_argument(
        "--source-std",
        type=float,
        default=1.0,
        help=(
            "source Gaussian std; 1 matches the latent GMD baseline, while 0.5 "
            "is the paper pixel-prior ablation"
        ),
    )
    parser.add_argument(
        "--coupling",
        choices=("independent", "teacher_coupled_ablation"),
        default="independent",
    )
    parser.add_argument("--gmd-teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--gmd-teacher-weight-kind", choices=("raw", "ema"), default="ema")
    parser.add_argument("--gmd-teacher-base-channels", type=int, default=64)
    parser.add_argument("--gmd-teacher-time-embedding-dim", type=int, default=32)
    parser.add_argument("--gmd-teacher-condition-dim", type=int, default=None)
    parser.add_argument(
        "--lsd-target",
        choices=("current", "ema_ablation"),
        default="current",
        help="current is the released stop-gradient target; EMA is a non-paper ablation",
    )

    parser.add_argument("--base-channels", type=int, default=256)
    parser.add_argument("--time-embedding-dim", type=int, default=32)
    parser.add_argument("--condition-dim", type=int, default=None)
    parser.add_argument(
        "--sigma-data",
        type=float,
        default=0.5,
        help="network-only EDM2 preconditioner x_in=x/sigma, phi=sigma*UNet(x_in)",
    )
    parser.add_argument(
        "--zero-init-velocity", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--logvar-channels", type=int, default=128)
    parser.add_argument("--use-bfloat16", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-decay-steps", type=int, default=35_000)
    parser.add_argument("--radam-beta1", type=float, default=0.9)
    parser.add_argument("--radam-beta2", type=float, default=0.999)
    parser.add_argument("--radam-eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.9999)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--distributed-timeout-minutes", type=int, default=30)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / "experiments" / "celeba_latent_lsd"
    )
    parser.add_argument("--run-label", default="latent_lsd")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=5_000)
    parser.add_argument(
        "--verify-gradients",
        action="store_true",
        help=(
            "after every backward, require every gradient to be finite and, "
            "under DDP, verify a rank-synchronized gradient fingerprint"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU/gloo run; works under torchrun and uses the cache prefix",
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


def validate_args(args: argparse.Namespace) -> None:
    if not args.cache.is_file():
        raise FileNotFoundError(f"latent cache not found: {args.cache}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.global_batch < 4 or args.global_batch % 4:
        raise ValueError("--global-batch must be a positive multiple of four")
    if args.global_batch % args.world_size:
        raise ValueError("--global-batch must be divisible by the torchrun world size")
    args.local_batch = args.global_batch // args.world_size
    if args.local_batch < 4 or args.local_batch % 4:
        raise ValueError("the local per-rank batch must be a positive multiple of four")
    if args.train_start_index < 0:
        raise ValueError("--train-start-index must be non-negative")
    if args.train_end_index is not None and args.train_end_index <= args.train_start_index:
        raise ValueError("--train-end-index must exceed --train-start-index")
    for name in ("source_std", "sigma_data", "lr", "radam_eps", "grad_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    if args.lr_decay_steps < 1:
        raise ValueError("--lr-decay-steps must be positive")
    if not 0 < args.radam_beta1 < 1 or not 0 < args.radam_beta2 < 1:
        raise ValueError("RAdam betas must lie strictly between zero and one")
    if not 0 < args.ema_decay < 1:
        raise ValueError("--ema-decay must lie strictly between zero and one")
    if args.logvar_channels < 4 or args.logvar_channels % 4:
        raise ValueError("--logvar-channels must be a positive multiple of four")
    if args.time_embedding_dim < 4 or args.time_embedding_dim % 4:
        raise ValueError("--time-embedding-dim must be a positive multiple of four")
    if args.log_every < 1 or args.checkpoint_every < 0:
        raise ValueError("logging must be positive and checkpoint interval non-negative")
    if args.distributed_timeout_minutes < 1:
        raise ValueError("--distributed-timeout-minutes must be positive")
    if args.coupling == "independent" and args.gmd_teacher_checkpoint is not None:
        raise ValueError("a GMD teacher is only valid for teacher_coupled_ablation")
    if args.coupling == "teacher_coupled_ablation":
        if args.gmd_teacher_checkpoint is None:
            raise ValueError("teacher_coupled_ablation requires --gmd-teacher-checkpoint")
        if not args.gmd_teacher_checkpoint.is_file():
            raise FileNotFoundError(
                f"GMD teacher checkpoint not found: {args.gmd_teacher_checkpoint}"
            )
        if args.source_std != 1.0:
            raise ValueError(
                "teacher_coupled_ablation currently requires --source-std 1 so its "
                "source matches the standard-prior GMD teacher"
            )


def setup_distributed(args: argparse.Namespace) -> None:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1
    if args.smoke:
        args.global_batch = min(args.global_batch, 4 * args.world_size)
        torch.set_num_threads(1)
    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested == "auto":
        requested = "cpu"
    if args.distributed:
        if requested == "cuda":
            if args.distributed_backend != "nccl":
                raise ValueError("GPU DDP requires the nccl backend (NCCL/RCCL)")
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA/ROCm requested but unavailable")
            torch.cuda.set_device(args.local_rank)
            args.device = torch.device("cuda", args.local_rank)
        elif requested == "cpu":
            if args.distributed_backend != "gloo":
                raise ValueError("CPU DDP requires --distributed-backend gloo")
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


def unwrap_model(model: nn.Module) -> CelebALatentLSDTrainingModel:
    result = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(result, CelebALatentLSDTrainingModel):
        raise TypeError("unexpected LSD training model wrapper")
    return result


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
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


class LatentBatchStream:
    """Deterministic global permutations with an exactly resumable position."""

    def __init__(self, latents: Tensor, args: argparse.Namespace) -> None:
        if latents.device.type != "cpu" or latents.ndim != 2:
            raise ValueError("stream latents must be a two-dimensional CPU tensor")
        self.latents = latents
        self.seed = int(args.seed)
        self.rank = int(args.rank)
        self.world_size = int(args.world_size)
        self.global_batch = int(args.global_batch)
        self.local_batch = int(args.local_batch)
        self.batches_per_epoch = latents.shape[0] // self.global_batch
        if self.batches_per_epoch < 1:
            raise ValueError("training split is smaller than one global batch")
        self.usable_count = self.batches_per_epoch * self.global_batch
        self.epoch = 0
        self.batch_offset = 0
        self._permutation = self._make_permutation()

    def _make_permutation(self) -> Tensor:
        generator = torch.Generator(device="cpu").manual_seed(self.seed + self.epoch)
        return torch.randperm(self.latents.shape[0], generator=generator)[: self.usable_count]

    def next(self) -> Tensor:
        if self.batch_offset == self.batches_per_epoch:
            self.epoch += 1
            self.batch_offset = 0
            self._permutation = self._make_permutation()
        start = self.batch_offset * self.global_batch + self.rank * self.local_batch
        indices = self._permutation[start : start + self.local_batch]
        self.batch_offset += 1
        return self.latents.index_select(0, indices)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_offset": self.batch_offset}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if set(state) != {"epoch", "batch_offset"}:
            raise ValueError("malformed data-stream state")
        epoch = int(state["epoch"])
        offset = int(state["batch_offset"])
        if epoch < 0 or not 0 <= offset <= self.batches_per_epoch:
            raise ValueError("invalid data-stream checkpoint position")
        self.epoch = epoch
        self.batch_offset = offset
        self._permutation = self._make_permutation()


def _strip_prefix(state: Mapping[str, Tensor], prefix: str) -> dict[str, Tensor]:
    if state and all(key.startswith(prefix) for key in state):
        return {key[len(prefix) :]: value for key, value in state.items()}
    return dict(state)


def _extract_teacher_state(
    payload: Any, weight_kind: str
) -> tuple[dict[str, Tensor], Mapping[str, Any] | None, str]:
    signature: Mapping[str, Any] | None = None
    if isinstance(payload, Mapping) and payload.get("format") == "driftmodel.celeba_latent_flow":
        signature = payload.get("training_signature")
        if not isinstance(signature, Mapping):
            raise ValueError("GMD training checkpoint lacks a training signature")
        if signature.get("method") != "drift" or signature.get("drift_objective") != "official_gmd":
            raise ValueError("teacher checkpoint is not attested as an official-GMD drift run")
        selected = payload.get("model") if weight_kind == "raw" else payload.get("ema")
        source = f"training_checkpoint_{weight_kind}"
    elif isinstance(payload, Mapping) and payload and all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in payload.items()
    ):
        selected = payload
        source = "unattested_exported_state"
    else:
        raise ValueError("unsupported GMD teacher checkpoint payload")
    if not isinstance(selected, Mapping):
        raise ValueError(f"GMD teacher checkpoint has no {weight_kind} state")
    state = {
        str(key): value
        for key, value in selected.items()
        if key != "n_averaged" and torch.is_tensor(value)
    }
    if not state:
        raise ValueError("GMD teacher state is empty")
    return state, signature, source


def load_gmd_teacher(
    args: argparse.Namespace,
    *,
    cache_sha256: str,
    train_start: int,
    train_end: int,
) -> tuple[nn.Module | None, dict[str, Any] | None]:
    if args.coupling == "independent":
        return None, None
    assert args.gmd_teacher_checkpoint is not None
    payload = torch.load(args.gmd_teacher_checkpoint, map_location="cpu", weights_only=True)
    state, signature, source = _extract_teacher_state(payload, args.gmd_teacher_weight_kind)
    if signature is not None:
        immutable_expected = {
            "cache_sha256": cache_sha256,
            "train_start_index": train_start,
            "train_end_index": train_end,
            "prior": "N(0,I)",
        }
        differing = [
            key
            for key, expected in immutable_expected.items()
            if signature.get(key, "N(0,I)" if key == "prior" else None) != expected
        ]
        if differing:
            raise ValueError(
                "GMD teacher coordinates/data disagree with this LSD run for "
                f"{differing}"
            )
        if signature.get("latent_normalization", {"mode": "none"}) not in (
            None,
            "none",
            {"mode": "none"},
        ):
            raise ValueError("GMD teacher must use unnormalized latent coordinates")
    base_channels = int(
        signature.get("base_channels", args.gmd_teacher_base_channels)
        if signature is not None
        else args.gmd_teacher_base_channels
    )
    time_embedding_dim = int(
        signature.get("time_embedding_dim", args.gmd_teacher_time_embedding_dim)
        if signature is not None
        else args.gmd_teacher_time_embedding_dim
    )
    condition_dim = (
        signature.get("condition_dim", args.gmd_teacher_condition_dim)
        if signature is not None
        else args.gmd_teacher_condition_dim
    )
    endpoint_parameterization = (
        signature.get("endpoint_parameterization", "absolute")
        if signature is not None
        else "absolute"
    )
    if endpoint_parameterization != "absolute":
        raise ValueError("teacher-coupled GMD requires an absolute-endpoint teacher")
    teacher = TransitionConditionedLatentUNet(
        base_channels=base_channels,
        time_embedding_dim=time_embedding_dim,
        condition_dim=condition_dim,
        zero_init_endpoint=False,
        endpoint_parameterization="absolute",
    )
    candidates: list[tuple[str, dict[str, Tensor]]] = [("identity", state)]
    without_module = _strip_prefix(state, "module.")
    candidates.extend(
        [
            ("strip_module", without_module),
            ("strip_flow_map", _strip_prefix(state, "flow_map.")),
            ("strip_module_flow_map", _strip_prefix(without_module, "flow_map.")),
        ]
    )
    target = teacher.state_dict()
    transform: str | None = None
    for name, candidate in candidates:
        if candidate.keys() == target.keys() and all(
            candidate[key].shape == target[key].shape for key in target
        ):
            teacher.load_state_dict(candidate, strict=True)
            transform = name
            break
    if transform is None:
        raise ValueError("GMD teacher weights do not exactly match the resolved architecture")
    teacher.to(args.device).eval().requires_grad_(False)
    metadata = {
        "path": str(args.gmd_teacher_checkpoint.resolve()),
        "sha256": sha256_file(args.gmd_teacher_checkpoint),
        "weight_kind": args.gmd_teacher_weight_kind,
        "source": source,
        "objective_attested": signature is not None,
        "load_transform": transform,
        "base_channels": base_channels,
        "time_embedding_dim": time_embedding_dim,
        "condition_dim": condition_dim,
        "endpoint_parameterization": endpoint_parameterization,
        "frozen": True,
        "coupling": "x1=stopgrad(T_GMD(x0,0,1)), same x0",
    }
    return teacher, metadata


def make_training_model(args: argparse.Namespace) -> CelebALatentLSDTrainingModel:
    backbone = CelebALatentLSDBackbone(
        base_channels=args.base_channels,
        time_embedding_dim=args.time_embedding_dim,
        condition_dim=args.condition_dim,
        sigma_data=args.sigma_data,
        zero_init_velocity=args.zero_init_velocity,
    )
    flow_map = ResidualTwoTimeFlowMap(
        backbone,
        MagnitudePreservingTimeLogVariance(args.logvar_channels),
    )
    model = CelebALatentLSDTrainingModel(flow_map)
    if project_log_variance_weight_(model) != 1:
        raise RuntimeError("log-variance projection did not find exactly one head")
    return model


def learning_rate_at_step(step_index: int, args: argparse.Namespace) -> float:
    return args.lr / math.sqrt(max(step_index / args.lr_decay_steps, 1.0))


def make_ema(model: CelebALatentLSDTrainingModel, decay: float) -> AveragedModel:
    ema = AveragedModel(
        model,
        multi_avg_fn=get_ema_multi_avg_fn(decay),
        use_buffers=True,
    )
    ema.n_averaged.fill_(1)  # theta_0 participates in the first EMA recursion.
    ema.eval().requires_grad_(False)
    return ema


def local_rng_state(args: argparse.Namespace) -> dict[str, Tensor]:
    state = {"cpu": torch.get_rng_state().cpu()}
    if args.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(args.device).cpu()
    return state


def gather_rng_states(args: argparse.Namespace) -> list[dict[str, Tensor]]:
    local = local_rng_state(args)
    if not args.distributed:
        return [local]
    gathered: list[dict[str, Tensor] | None] = [None] * args.world_size
    dist.all_gather_object(gathered, local)
    if any(item is None for item in gathered):
        raise RuntimeError("failed to gather every rank RNG state")
    return [item for item in gathered if item is not None]


def restore_rng_state(states: Any, args: argparse.Namespace) -> None:
    if not isinstance(states, list) or len(states) != args.world_size:
        raise ValueError("checkpoint RNG world size does not match this launch")
    state = states[args.rank]
    if not isinstance(state, Mapping) or not torch.is_tensor(state.get("cpu")):
        raise ValueError("checkpoint has malformed CPU RNG state")
    torch.set_rng_state(state["cpu"].cpu())
    if args.device.type == "cuda":
        if not torch.is_tensor(state.get("cuda")):
            raise ValueError("checkpoint lacks CUDA/ROCm RNG state")
        torch.cuda.set_rng_state(state["cuda"].cpu(), args.device)


def reduce_stats(stats: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Tensor]:
    keys: list[str] = []
    values: list[Tensor] = []
    for key, value in stats.items():
        tensor = (
            value.detach().to(device=args.device, dtype=torch.float32)
            if torch.is_tensor(value)
            else torch.tensor(float(value), device=args.device)
        )
        if tensor.numel() == 1:
            keys.append(key)
            values.append(tensor.reshape(()))
    packed = torch.stack(values)
    if args.distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(args.world_size)
    return {key: packed[index] for index, key in enumerate(keys)}


def training_signature(
    args: argparse.Namespace,
    cache_metadata: Mapping[str, Any],
    *,
    cache_sha256: str,
    train_values_sha256: str,
    train_start: int,
    train_end: int,
    teacher_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "signature_version": SIGNATURE_VERSION,
        "objective": "mixed_diagonal_fm_and_lagrangian_self_distillation",
        "flow_parameterization": "X=x+(t-s)*phi",
        "endpoint_parameterization": "absolute",
        "time_conditioning": "start_delta_v1",
        "diagonal_fraction": DIAGONAL_FRACTION,
        "diagonal_count_local": 3 * args.local_batch // 4,
        "offdiagonal_count_local": args.local_batch // 4,
        "offdiagonal_time_sampling": "ordered_two_independent_uniforms",
        "coordinate_reduction": "sum_then_batch_mean",
        "interpolant": "linear",
        "coupling": args.coupling,
        "lsd_target": args.lsd_target,
        "source_distribution": f"N(0,{args.source_std ** 2:g}I)",
        "source_std": args.source_std,
        "base_channels": args.base_channels,
        "time_embedding_dim": args.time_embedding_dim,
        "condition_dim": args.condition_dim,
        "sigma_data": args.sigma_data,
        "zero_init_velocity": args.zero_init_velocity,
        "logvar_channels": args.logvar_channels,
        "use_bfloat16": args.use_bfloat16,
        "optimizer": "RAdam",
        "lr": args.lr,
        "lr_schedule": "sqrt_after_decay_steps",
        "lr_decay_steps": args.lr_decay_steps,
        "radam_beta1": args.radam_beta1,
        "radam_beta2": args.radam_beta2,
        "radam_eps": args.radam_eps,
        "grad_clip": args.grad_clip,
        "ema_decay": args.ema_decay,
        "ema_is_lsd_target": args.lsd_target == "ema_ablation",
        "edm2_projection": "logvar_weight_only_after_every_update",
        "ddp_broadcast_buffers": False,
        "ddp_boundary": "one outer forward containing velocity JVP, current target, and logvar",
        "global_batch": args.global_batch,
        "local_batch": args.local_batch,
        "world_size": args.world_size,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "verify_gradients": args.verify_gradients,
        "latent_dim": CELEBA_LATENT_DIM,
        "latent_normalization": "none",
        "cache_model_id": cache_metadata.get("model_id"),
        "cache_scaling_factor": cache_metadata.get("scaling_factor"),
        "cache_encoding": cache_metadata.get("encoding"),
        "cache_sha256": cache_sha256,
        "train_values_sha256": train_values_sha256,
        "train_start_index": train_start,
        "train_end_index": train_end,
        "teacher_checkpoint_sha256": (
            teacher_metadata.get("sha256") if teacher_metadata is not None else None
        ),
        "teacher_weight_kind": (
            teacher_metadata.get("weight_kind") if teacher_metadata is not None else None
        ),
        "teacher_load_transform": (
            teacher_metadata.get("load_transform") if teacher_metadata is not None else None
        ),
        "source_files_sha256": _source_hashes(),
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
                raise ValueError("--run-dir disagrees with the directory implied by --resume")
            path = inferred
            if not path.is_dir():
                raise FileNotFoundError(f"resume run directory is missing: {path}")
        elif args.run_dir is not None:
            path = args.run_dir.resolve()
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"new run directory is not empty: {path}")
            path.mkdir(parents=True, exist_ok=True)
        else:
            label = "".join(
                char if char.isalnum() or char in "-_" else "_" for char in args.run_label
            ).strip("_")
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = (args.output_root / f"{stamp}_{label}").resolve()
            path.mkdir(parents=True, exist_ok=False)
        return str(path)

    return Path(run_on_rank_zero(args, resolve))


def save_checkpoint(
    path: Path,
    *,
    completed_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: AveragedModel,
    stream: LatentBatchStream,
    signature: Mapping[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    rng_states = gather_rng_states(args)

    def save() -> None:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "completed_step": completed_step,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict(),
            "data_stream": stream.state_dict(),
            "rng_by_rank": rng_states,
            "training_signature": dict(signature),
            "run_dir": str(run_dir.resolve()),
            "saved_at_utc": utc_now(),
        }
        atomic_torch_save(path, payload)

    run_on_rank_zero(args, save)


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: AveragedModel,
    stream: LatentBatchStream,
    signature: Mapping[str, Any],
    run_dir: Path,
    args: argparse.Namespace,
) -> int:
    checkpoint = torch.load(path, map_location=args.device, weights_only=True)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("--resume expects a CelebA latent LSD checkpoint")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported LSD checkpoint version: {checkpoint.get('version')}")
    if checkpoint.get("training_signature") != dict(signature):
        found = checkpoint.get("training_signature")
        found = found if isinstance(found, Mapping) else {}
        differing = sorted(
            key
            for key in set(found).union(signature)
            if found.get(key) != signature.get(key)
        )
        raise ValueError(f"resume training signature differs for {differing}")
    saved_run_dir = checkpoint.get("run_dir")
    if not isinstance(saved_run_dir, str) or Path(saved_run_dir).resolve() != run_dir.resolve():
        raise ValueError("resume checkpoint run directory provenance does not match")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    ema.load_state_dict(checkpoint["ema"])
    stream.load_state_dict(checkpoint["data_stream"])
    restore_rng_state(checkpoint["rng_by_rank"], args)
    completed = int(checkpoint["completed_step"])
    if completed < 0:
        raise ValueError("checkpoint has a negative completed step")
    return completed


def write_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    cache_metadata: Mapping[str, Any],
    selected: Tensor,
    train_start: int,
    train_end: int,
    signature: Mapping[str, Any],
    model: CelebALatentLSDTrainingModel,
    teacher_metadata: Mapping[str, Any] | None,
) -> None:
    suffix = "" if args.resume is None else f"_resume_{utc_now().replace(':', '-')}"
    snapshot = archive_source_snapshot(run_dir, f"source_snapshot{suffix}")
    config = {
        **{key: value for key, value in vars(args).items() if not key.startswith("_")},
        "train_start_resolved": train_start,
        "train_end_resolved": train_end,
        "train_count": selected.shape[0],
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "architecture": model.flow_map.velocity_network.architecture_config(),
        "training_signature": signature,
        "paper_reference": {
            "repository_commit": OFFICIAL_LSD_COMMIT,
            "pixel_prior_source_std": 0.5,
            "pixel_global_batch_reported_in_paper": 256,
            "pixel_updates_reported_in_paper": 800_000,
            "public_config_global_batch": 512,
            "public_config_updates": 400_000,
            "total_samples_both": 204_800_000,
        },
        "adaptation_notice": (
            "LSD objective port to frozen-VAE 4x8x8 latents and a compact ordinary-convolution "
            "U-Net; not the paper's pixel-space 124M EDM2 architecture or FID protocol."
        ),
        "initialization_notice": (
            "phi is the exact zero function initially, matching released EDM2 behavior; "
            "this adaptation zeroes output_conv instead of EDM2's scalar out_gain."
        ),
    }
    atomic_json_dump(run_dir / f"config{suffix}.json", config)
    provenance = {
        "created_at_utc": utc_now(),
        "command": args.command,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "hip_runtime": getattr(torch.version, "hip", None),
        "device": str(args.device),
        "world_size": args.world_size,
        "global_batch": args.global_batch,
        "git": _git_provenance(),
        "source_snapshot": snapshot,
        "official_lsd_source_commit": OFFICIAL_LSD_COMMIT,
        "cache": {
            "path": str(args.cache.resolve()),
            "sha256": signature["cache_sha256"],
            "metadata": cache_metadata,
            "train_values_sha256": signature["train_values_sha256"],
            "selected_mean": float(selected.mean()),
            "selected_std": float(selected.std()),
            "selected_rms": float(selected.square().mean().sqrt()),
        },
        "teacher_coupled_ablation": teacher_metadata,
        "edm2_projection": {
            "rule": "RMS-unit projection with epsilon 1e-4 after initialization and every update",
            "scope": "MagnitudePreservingTimeLogVariance.weight only",
            "limitation": "compact latent U-Net convolutions are not MPConv",
        },
        "initialization": {
            "initial_phi": "exact_zero_function" if args.zero_init_velocity else "random_ablation",
            "official_mechanism": "zero scalar out_gain",
            "adapted_mechanism": "zero output convolution",
        },
        "ddp": {
            "broadcast_buffers": False,
            "reason": (
                "Fourier buffers are deterministic and identical; rebroadcasting them "
                "between repeated JVP forwards can invalidate autograd version counters."
            ),
            "loss_inside_single_outer_forward": True,
        },
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_NODELIST", "SLURM_PROCID")
        },
    }
    atomic_json_dump(run_dir / f"provenance{suffix}.json", provenance)


def _select_training_latents(
    latents: Tensor, args: argparse.Namespace
) -> tuple[Tensor, Tensor, int, int]:
    start = args.train_start_index
    end = latents.shape[0] if args.train_end_index is None else args.train_end_index
    if args.smoke:
        end = min(end, start + max(2 * args.global_batch, 8))
    if start >= latents.shape[0] or end > latents.shape[0] or end <= start:
        raise ValueError(
            f"requested training indices [{start},{end}) outside cache of {latents.shape[0]}"
        )
    spatial = latents[start:end].detach().cpu().contiguous()
    flat = flatten_latents(spatial).to(dtype=torch.float32).contiguous()
    if flat.shape[0] < args.global_batch:
        raise ValueError("selected training split is smaller than --global-batch")
    return spatial, flat, start, end


def _teacher_coupled_targets(teacher: nn.Module, x0: Tensor) -> Tensor:
    times = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
    with torch.no_grad():
        return teacher(x0, times, torch.ones_like(times)).detach()


def verify_smoke_gradients(model: nn.Module, args: argparse.Namespace) -> None:
    """Prove finite, synchronized gradients in smoke or audited GPU runs."""

    if not (args.smoke or args.verify_gradients):
        return
    parameters = tuple(unwrap_model(model).parameters())
    missing = sum(parameter.grad is None for parameter in parameters)
    if missing:
        raise RuntimeError(f"smoke backward left {missing} parameter gradients missing")
    if any(not bool(torch.isfinite(parameter.grad).all()) for parameter in parameters):
        raise FloatingPointError("smoke backward produced non-finite gradients")
    fingerprint = torch.stack(
        (
            sum(parameter.grad.float().sum() for parameter in parameters),
            sum(parameter.grad.float().square().sum() for parameter in parameters),
        )
    )
    if args.distributed:
        minimum = fingerprint.clone()
        maximum = fingerprint.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if not torch.allclose(minimum, maximum, rtol=1e-5, atol=1e-6):
            raise RuntimeError("DDP smoke gradients are not synchronized across ranks")


def _weights_payload(
    *,
    kind: str,
    state: Mapping[str, Tensor],
    completed_step: int,
    signature: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format": WEIGHTS_FORMAT,
        "version": WEIGHTS_VERSION,
        "kind": kind,
        "completed_step": completed_step,
        "model": dict(state),
        "training_signature": dict(signature),
        "architecture": dict(architecture),
        "saved_at_utc": utc_now(),
    }


def train(args: argparse.Namespace) -> Path:
    cache_latents, cache_metadata = load_latent_cache(args.cache)
    selected_spatial, training_latents, train_start, train_end = _select_training_latents(
        cache_latents, args
    )
    cache_digest = sha256_file(args.cache)
    values_digest = sha256_tensor(selected_spatial)
    stream = LatentBatchStream(training_latents, args)

    seed_initial_parameters(args)
    model = make_training_model(args).to(args.device)
    if model.flow_map.velocity_network.endpoint_parameterization != "absolute":
        raise RuntimeError("LSD backbone must expose absolute phi through endpoint()")
    raw_model: nn.Module = model
    if args.distributed:
        ddp_kwargs: dict[str, Any] = {"broadcast_buffers": False}
        if args.device.type == "cuda":
            ddp_kwargs.update(device_ids=[args.local_rank], output_device=args.local_rank)
        raw_model = DistributedDataParallel(model, **ddp_kwargs)

    teacher, teacher_metadata = load_gmd_teacher(
        args,
        cache_sha256=cache_digest,
        train_start=train_start,
        train_end=train_end,
    )
    # Stochastic streams become rank-specific only after every model has been
    # constructed and DDP has performed its initial parameter broadcast.
    torch.manual_seed(args.seed + args.rank)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed + args.rank)
    optimizer = torch.optim.RAdam(
        raw_model.parameters(),
        lr=args.lr,
        betas=(args.radam_beta1, args.radam_beta2),
        eps=args.radam_eps,
        weight_decay=0.0,
    )
    ema = make_ema(unwrap_model(raw_model), args.ema_decay)
    signature = training_signature(
        args,
        cache_metadata,
        cache_sha256=cache_digest,
        train_values_sha256=values_digest,
        train_start=train_start,
        train_end=train_end,
        teacher_metadata=teacher_metadata,
    )
    run_dir = resolve_run_directory(args)
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(
            args.resume,
            model=raw_model,
            optimizer=optimizer,
            ema=ema,
            stream=stream,
            signature=signature,
            run_dir=run_dir,
            args=args,
        )
    if start_step >= args.steps:
        raise ValueError(
            f"checkpoint is at step {start_step}; --steps must be larger for a safe resume"
        )

    run_on_rank_zero(
        args,
        lambda: write_metadata(
            args,
            run_dir,
            cache_metadata,
            training_latents,
            train_start,
            train_end,
            signature,
            unwrap_model(raw_model),
            teacher_metadata,
        ),
    )
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    if is_main_process(args):
        logger.log(
            {
                "event": "resume" if start_step else "start",
                "step": start_step,
                "run_dir": str(run_dir),
                "train_count": training_latents.shape[0],
                "global_batch": args.global_batch,
                "local_batch": args.local_batch,
                "coupling": args.coupling,
                "lsd_target": args.lsd_target,
                "source_std": args.source_std,
                "model_parameters": sum(p.numel() for p in unwrap_model(raw_model).parameters()),
            }
        )

    started_at = time.monotonic()
    progress = trange(
        start_step,
        args.steps,
        initial=start_step,
        total=args.steps,
        desc="celeba-latent:lsd",
        disable=not is_main_process(args),
    )
    for step_index in progress:
        learning_rate = learning_rate_at_step(step_index, args)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        x1_data = stream.next().to(args.device, non_blocking=args.device.type == "cuda")
        x0 = args.source_std * torch.randn_like(x1_data)
        x1 = x1_data if teacher is None else _teacher_coupled_targets(teacher, x0)
        unwrapped = unwrap_model(raw_model)
        target_flow = (
            unwrapped.flow_map
            if args.lsd_target == "current"
            else ema.module.flow_map
        )
        autocast_enabled = args.use_bfloat16 and args.device.type == "cuda"
        with torch.autocast(
            device_type=args.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            loss, local_stats = raw_model(x0, x1, target_flow)
        if local_stats["diagonal_count"] != 3 * args.local_batch // 4:
            raise RuntimeError("mixed loss did not preserve the exact 75/25 local split")
        if local_stats["offdiagonal_count"] != args.local_batch // 4:
            raise RuntimeError("mixed loss did not preserve the exact 75/25 local split")
        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        if args.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"non-finite LSD loss at step {step_index + 1}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        verify_smoke_gradients(raw_model, args)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), args.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        projected_count = project_log_variance_weight_(unwrap_model(raw_model))
        if projected_count != 1:
            raise RuntimeError("unexpected number of projected logvar heads")
        ema.update_parameters(unwrap_model(raw_model))
        completed_step = step_index + 1

        should_log = (
            completed_step == 1
            or completed_step == args.steps
            or completed_step % args.log_every == 0
        )
        if should_log:
            reduced = reduce_stats(
                {
                    **local_stats,
                    "grad_norm": grad_norm,
                    "source_rms": x0.detach().square().mean().sqrt(),
                    "target_endpoint_rms": x1.detach().square().mean().sqrt(),
                    "data_rms": x1_data.detach().square().mean().sqrt(),
                },
                args,
            )
            record = {
                "event": "train",
                "step": completed_step,
                "epoch": stream.epoch,
                "batch_offset": stream.batch_offset,
                "elapsed_seconds": time.monotonic() - started_at,
                "learning_rate": learning_rate,
                **{key: float(value.cpu()) for key, value in reduced.items()},
            }
            if is_main_process(args):
                logger.log(record)
                progress.set_postfix(loss=f"{record['loss']:.3e}")

        if args.checkpoint_every and completed_step % args.checkpoint_every == 0:
            save_checkpoint(
                run_dir / "checkpoints" / f"checkpoint_step_{completed_step:08d}.pt",
                completed_step=completed_step,
                model=raw_model,
                optimizer=optimizer,
                ema=ema,
                stream=stream,
                signature=signature,
                run_dir=run_dir,
                args=args,
            )

    save_checkpoint(
        run_dir / "checkpoint_final.pt",
        completed_step=args.steps,
        model=raw_model,
        optimizer=optimizer,
        ema=ema,
        stream=stream,
        signature=signature,
        run_dir=run_dir,
        args=args,
    )

    def save_exports() -> None:
        unwrapped = unwrap_model(raw_model)
        architecture = unwrapped.flow_map.velocity_network.architecture_config()
        atomic_torch_save(
            run_dir / "model_raw_final.pt",
            _weights_payload(
                kind="raw",
                state=unwrapped.state_dict(),
                completed_step=args.steps,
                signature=signature,
                architecture=architecture,
            ),
        )
        atomic_torch_save(
            run_dir / "model_ema_final.pt",
            _weights_payload(
                kind="ema",
                state=ema.module.state_dict(),
                completed_step=args.steps,
                signature=signature,
                architecture=architecture,
            ),
        )
        logger.log(
            {
                "event": "complete",
                "step": args.steps,
                "elapsed_seconds": time.monotonic() - started_at,
                "run_dir": str(run_dir),
            }
        )

    run_on_rank_zero(args, save_exports)
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
