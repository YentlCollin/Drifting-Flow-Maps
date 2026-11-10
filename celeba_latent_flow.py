"""Train CelebA WGF maps in the 4x8x8 latent space of a frozen VAE.

The experiment keeps the MNIST theory unchanged. Particles are ordinary
vectors in R^256 and the existing WGF losses operate on those vectors. The
default prior remains N(0, I); an explicitly selected train-only full Gaussian
is available for the exploratory prior/initialization ablation. The VAE is
used only to construct the deterministic data cache and decode diagnostics.

Typical distributed launch (``nccl`` is also the backend name used by RCCL)::

    torchrun --standalone --nproc_per_node=8 celeba_latent_flow.py \
      --cache data/celeba64_latents_sdvae_ft_mse.pt \
      --vae-model data/models/sd-vae-ft-mse \
      --method flow --global-kde-support --synchronize-times

A dependency-light CPU preflight does not load diffusers::

    python celeba_latent_flow.py --smoke --no-decode-samples
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil


def _configure_rank_local_miopen_cache() -> None:
    """Avoid shared read-only/locked MIOpen SQLite caches under torchrun."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    cache_base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
    if cache_base is None and world_size > 1:
        job_token = os.environ.get("SLURM_JOB_ID", os.environ.get("MASTER_PORT", "local"))
        cache_base = f"/tmp/driftmodel_miopen_{job_token}"
    if cache_base is None:
        return
    rank_directory = Path(cache_base) / f"rank_{os.environ.get('LOCAL_RANK', '0')}"
    for directory in (rank_directory, rank_directory / "userdb", rank_directory / "kernels"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MIOPEN_CACHE_DIR", str(rank_directory))
    os.environ.setdefault("MIOPEN_USER_DB_PATH", str(rank_directory / "userdb"))
    os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(rank_directory / "kernels"))


_configure_rank_local_miopen_cache()

import argparse
import hashlib
import importlib.util
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
import warnings
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import trange

from local_paths import REPO_ROOT
from src.celeba_latent import (
    CELEBA_LATENT_DIM,
    LATENT_PRIOR_FILENAME,
    CelebALatentCodec,
    LatentAffineNormalization,
    LatentGaussianPrior,
    LatentGaussianSampler,
    TransitionConditionedLatentUNet,
    directory_file_hashes,
    flatten_latents,
    load_latent_cache,
    sha256_tensor,
    unflatten_latents,
)
from src.wgf import (
    fast_wgf_flow_map_loss,
    optimizer_time_drifting_loss,
    sample_wgf_flow_map,
    wgf_velocity_diagnostics,
)
from src.official_drifting import (
    OFFICIAL_TEMPERATURES,
    raw_latent_pyramid_features,
    summed_feature_drifting_loss,
)


CHECKPOINT_FORMAT = "driftmodel.celeba_latent_flow"
CHECKPOINT_VERSION = 1
TRAINING_SIGNATURE_VERSION = 2
SAMPLE_NFES = (1, 2, 4, 8, 16, 50)
DEFAULT_TRANSITION_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() != 1:
            return value.detach().cpu().tolist()
        value = value.detach().cpu().item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
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
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(_json_safe(payload), temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
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
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        payload = {"timestamp_utc": utc_now(), **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_json_safe(payload), sort_keys=True, allow_nan=False) + "\n")
            handle.flush()


def _run_git(*arguments: str, text: bool = True) -> str | bytes | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    relative_paths = (
        "celeba_latent_flow.py",
        "celeba_latent_mae.py",
        "src/celeba_latent.py",
        "src/celeba_gmd_features.py",
        "src/celeba_gmd_features_v2.py",
        "src/official_drifting.py",
        "src/wgf.py",
        "pyproject.toml",
    )
    return {
        relative_path: sha256_file(REPO_ROOT / relative_path)
        for relative_path in relative_paths
        if (REPO_ROOT / relative_path).is_file()
    }


def archive_source_snapshot(run_dir: Path, name: str = "source_snapshot") -> dict[str, Any]:
    """Store reconstructible source bytes, not only hashes, with a run."""

    snapshot_dir = run_dir / name
    if snapshot_dir.exists():
        raise FileExistsError(f"source snapshot already exists: {snapshot_dir}")
    before = source_hashes()
    for relative_path in before:
        source = REPO_ROOT / relative_path
        destination = snapshot_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    after = source_hashes()
    if after != before:
        raise RuntimeError("training sources changed while their snapshot was being archived")
    copied = {
        relative_path: sha256_file(snapshot_dir / relative_path)
        for relative_path in before
    }
    if copied != before:
        raise RuntimeError("archived training sources do not match their source hashes")
    manifest = {
        "format": "driftmodel.source_snapshot",
        "version": 1,
        "created_at_utc": utc_now(),
        "source_sha256": copied,
    }
    atomic_json_dump(snapshot_dir / "manifest.json", manifest)
    return {
        "path": str(snapshot_dir),
        "manifest_sha256": sha256_file(snapshot_dir / "manifest.json"),
        **manifest,
    }


def git_provenance() -> dict[str, Any]:
    diff = _run_git("diff", "--binary", "HEAD", text=False)
    return {
        "commit": (_run_git("rev-parse", "HEAD") or "unavailable").strip(),
        "branch": (_run_git("rev-parse", "--abbrev-ref", "HEAD") or "unavailable").strip(),
        "status_porcelain": (_run_git("status", "--short") or "unavailable").splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest() if isinstance(diff, bytes) else "unavailable",
        "source_sha256": source_hashes(),
    }


def _parse_optional_bandwidth(value: str) -> float | None:
    if value.lower() in {"auto", "median", "none"}:
        return None
    bandwidth = float(value)
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise argparse.ArgumentTypeError("bandwidth must be positive or one of auto/median/none")
    return bandwidth


def _parse_transition_times(value: str) -> tuple[float, ...]:
    try:
        times = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("transition times must be comma-separated floats") from error
    if len(times) < 2 or any(not math.isfinite(item) for item in times):
        raise argparse.ArgumentTypeError("at least two finite transition times are required")
    if times[0] != 0.0 or times[-1] != 1.0 or any(left >= right for left, right in zip(times, times[1:])):
        raise argparse.ArgumentTypeError("transition times must increase strictly from 0 to 1")
    return times


def _parse_temperatures(value: str) -> tuple[float, ...]:
    try:
        temperatures = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "temperatures must be comma-separated floats"
        ) from error
    if not temperatures or any(
        not math.isfinite(temperature) or temperature <= 0
        for temperature in temperatures
    ):
        raise argparse.ArgumentTypeError(
            "temperatures must be finite and strictly positive"
        )
    return temperatures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CelebA fast WGF flow map in deterministic SD-VAE latent space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=("flow", "drift"), default="flow")
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "data" / "celeba64_latents_sdvae_ft_mse.pt",
        help="posterior-mean cache produced by src.celeba_latent",
    )
    parser.add_argument(
        "--train-start-index",
        type=int,
        default=5_000,
        help="inclusive cache index; the canonical CelebA split reserves indices 0:5000",
    )
    parser.add_argument(
        "--train-end-index",
        type=int,
        default=30_000,
        help="exclusive cache index",
    )
    parser.add_argument(
        "--latent-normalization",
        choices=("none", "channel"),
        default="none",
        help="affine channel standardization fitted only on the resolved training split",
    )
    parser.add_argument(
        "--latent-prior",
        choices=("standard", "train-full-gaussian"),
        default="standard",
        help="source distribution; fitted full Gaussian uses only the resolved training split",
    )
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch", type=int, default=128, help="local batch size per rank")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-embedding-dim", type=int, default=32)
    parser.add_argument("--condition-dim", type=int, default=None)
    parser.add_argument("--zero-init-endpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--endpoint-parameterization",
        choices=("absolute", "residual"),
        default="absolute",
        help="absolute endpoint g_theta or source-plus-residual endpoint x+r_theta",
    )

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-warmup-steps", type=int, default=500)
    parser.add_argument(
        "--lr-warmup-steps",
        type=int,
        default=0,
        help="linear learning-rate warmup; the released latent drifting recipe uses 5000",
    )

    parser.add_argument("--velocity", choices=("kl", "mmd"), default="kl")
    parser.add_argument(
        "--drift-objective",
        choices=("kde_wgf", "official_gmd"),
        default="kde_wgf",
        help=(
            "kde_wgf preserves the historical Gaussian score-difference baseline; "
            "official_gmd uses the released row/column-normalized Laplacian drifting loss"
        ),
    )
    parser.add_argument(
        "--drift-features",
        "--gmd-features",
        dest="drift_features",
        choices=("raw_pyramid", "mae"),
        default="raw_pyramid",
        help=(
            "feature families used by official_gmd, either as the direct drifting "
            "objective or as the flow-map endpoint; mae requires --feature-checkpoint"
        ),
    )
    parser.add_argument(
        "--feature-checkpoint",
        type=Path,
        default=None,
        help="frozen CelebA latent MAE checkpoint for --drift-features mae",
    )
    parser.add_argument(
        "--drift-temperatures",
        "--gmd-temperatures",
        dest="drift_temperatures",
        type=_parse_temperatures,
        default=OFFICIAL_TEMPERATURES,
        help="comma-separated Laplacian temperatures used by official_gmd",
    )
    parser.add_argument(
        "--drift-global-statistics",
        "--gmd-global-statistics",
        dest="drift_global_statistics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="share official loss scale/RMS statistics across DDP ranks",
    )
    parser.add_argument(
        "--drift-global-particle-support",
        "--gmd-global-particle-support",
        dest="drift_global_particle_support",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use positives and detached generated targets from every DDP rank; "
            "local queries retain gradients"
        ),
    )
    parser.add_argument(
        "--flow-boundary-objective",
        choices=("kde_wgf", "official_gmd"),
        default="kde_wgf",
        help=(
            "terminal distribution objective for --method flow; official_gmd "
            "replaces the raw-KDE origin boundary by the released feature-space loss"
        ),
    )
    parser.add_argument(
        "--lambda-feature-boundary",
        type=float,
        default=0.003,
        help=(
            "weight of the official GMD endpoint loss when "
            "--flow-boundary-objective official_gmd"
        ),
    )
    parser.add_argument(
        "--bandwidth",
        type=_parse_optional_bandwidth,
        default=3.0,
        help="fixed RBF width; use auto/median for the batch estimator",
    )
    parser.add_argument(
        "--rbf-geometry",
        choices=("raw", "normalized", "dimension_normalized"),
        default="raw",
    )
    parser.add_argument("--global-kde-support", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--synchronize-times", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--time-groups", type=int, default=4)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--schedule", choices=("constant", "rational"), default="constant")
    parser.add_argument("--time-scale", type=float, default=100.0)
    parser.add_argument("--schedule-eps", type=float, default=1e-3)
    parser.add_argument("--path-weight", type=float, default=0.1)
    parser.add_argument("--lambda-wgf", type=float, default=0.25)
    parser.add_argument("--lambda-eulerian", type=float, default=0.0)
    parser.add_argument(
        "--lambda-semigroup",
        type=float,
        default=0.1,
        help=(
            "weight of the canonical composition anchor on model-reached states: "
            "Phi(Phi(z,0,a),a,b) = Phi(z,0,b)"
        ),
    )
    parser.add_argument(
        "--lambda-terminal-semigroup",
        type=float,
        default=0.0,
        help=(
            "weight of endpoint composition consistency: "
            "Phi(stopgrad(Phi(z,0,a)),a,1) = stopgrad(Phi(z,0,1))"
        ),
    )
    parser.add_argument("--lambda-lagrangian", type=float, default=0.0)
    parser.add_argument("--lambda-boundary-origin", type=float, default=5.0)
    parser.add_argument("--lambda-boundary-arbitrary", type=float, default=0.0)
    parser.add_argument("--boundary-step-size", type=float, default=1.0)
    parser.add_argument("--distill-min-span", type=float, default=0.25)
    parser.add_argument("--distill-max-span", type=float, default=0.5)
    parser.add_argument("--drift-step-size", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--distributed-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--distributed-timeout-minutes", type=int, default=30)

    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs" / "celeba_latent_wgf")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--initial-model-checkpoint",
        type=Path,
        default=None,
        help=(
            "weights-only initialization, e.g. an endpoint GMD checkpoint before "
            "flow-map consistency training; cannot be combined with --resume"
        ),
    )
    parser.add_argument(
        "--initial-weight-kind",
        choices=("raw", "ema"),
        default="ema",
        help="state to extract when --initial-model-checkpoint is a training checkpoint",
    )
    parser.add_argument(
        "--endpoint-teacher-checkpoint",
        type=Path,
        default=None,
        help=(
            "frozen endpoint teacher used during flow-map distillation; normally the "
            "same trained GMD endpoint used for weights-only initialization"
        ),
    )
    parser.add_argument(
        "--endpoint-teacher-weight-kind",
        choices=("raw", "ema"),
        default="ema",
    )
    parser.add_argument(
        "--lambda-endpoint-teacher",
        type=float,
        default=0.0,
        help="MSE weight keeping Phi(z,0,1) close to a frozen endpoint teacher",
    )
    parser.add_argument("--allow-resume-config-mismatch", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--checkpoint-every", type=int, default=1_000, help="0 disables periodic checkpoints")
    parser.add_argument("--sample-every", type=int, default=1_000, help="0 disables periodic sampling")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--sample-time-power", type=float, default=1.0)
    parser.add_argument("--diagnostic-count", type=int, default=64)
    parser.add_argument(
        "--transition-times",
        type=_parse_transition_times,
        default=DEFAULT_TRANSITION_TIMES,
    )
    parser.add_argument("--save-latent-samples", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--decode-samples", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae-model", type=Path, default=REPO_ROOT / "data" / "models" / "sd-vae-ft-mse")
    parser.add_argument("--vae-scaling-factor", type=float, default=None)
    parser.add_argument("--vae-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--decode-batch", type=int, default=16)
    parser.add_argument("--smoke", action="store_true", help="force a tiny CPU run without loading diffusers")
    return parser


def apply_smoke_overrides(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    args.device = "cpu"
    args.distributed_backend = "gloo"
    args.steps = min(args.steps, 2)
    args.batch = min(args.batch, 4)
    args.base_channels = min(args.base_channels, 8)
    args.time_embedding_dim = min(args.time_embedding_dim, 8)
    args.condition_dim = 16 if args.condition_dim is None else min(args.condition_dim, 16)
    args.time_groups = 1
    args.num_workers = 0
    args.pin_memory = False
    args.global_kde_support = False
    args.synchronize_times = False
    args.decode_samples = False
    args.sample_every = 0
    args.sample_count = min(args.sample_count, 4)
    args.diagnostic_count = min(args.diagnostic_count, 8)
    args.log_every = 1
    args.checkpoint_every = 0
    args.ema_warmup_steps = 0
    args.lr_warmup_steps = 0
    args.drift_features = "raw_pyramid"
    args.feature_checkpoint = None


def validate_args(args: argparse.Namespace) -> None:
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.batch < 2:
        raise ValueError("--batch must be at least 2 for particle KDE")
    if args.time_groups < 1 or args.batch % args.time_groups:
        raise ValueError("--time-groups must be positive and divide the local batch")
    if not 0.0 < args.dt < 0.5:
        raise ValueError("--dt must lie strictly between 0 and 0.5")
    if not 0.0 < args.time_scale:
        raise ValueError("--time-scale must be positive")
    if not 0.0 < args.schedule_eps:
        raise ValueError("--schedule-eps must be positive")
    if not 0.0 < args.distill_min_span <= args.distill_max_span <= 1.0:
        raise ValueError("distillation spans must satisfy 0 < min <= max <= 1")
    for name in (
        "path_weight",
        "lambda_wgf",
        "lambda_eulerian",
        "lambda_semigroup",
        "lambda_terminal_semigroup",
        "lambda_lagrangian",
        "lambda_boundary_origin",
        "lambda_boundary_arbitrary",
        "lambda_feature_boundary",
        "lambda_endpoint_teacher",
        "boundary_step_size",
        "drift_step_size",
        "weight_decay",
        "grad_clip",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.method == "flow" and not any(
        (
            args.path_weight * args.lambda_wgf,
            args.path_weight * args.lambda_eulerian,
            args.path_weight * args.lambda_lagrangian,
            args.lambda_semigroup,
            args.lambda_terminal_semigroup,
            args.lambda_boundary_origin,
            args.lambda_boundary_arbitrary,
            (
                args.lambda_feature_boundary
                if args.flow_boundary_objective == "official_gmd"
                else 0.0
            ),
            args.lambda_endpoint_teacher,
        )
    ):
        raise ValueError("at least one flow-map loss term must be active")
    if args.lr <= 0 or not 0 < args.adam_beta1 < 1 or not 0 < args.adam_beta2 < 1:
        raise ValueError("invalid optimizer hyperparameters")
    if (
        not 0.0 <= args.ema_decay < 1.0
        or args.ema_warmup_steps < 0
        or args.lr_warmup_steps < 0
    ):
        raise ValueError("EMA decay must be in [0,1) and warmup non-negative")
    if args.method != "drift" and args.drift_objective != "kde_wgf":
        raise ValueError("--drift-objective only applies to --method drift")
    if args.method != "flow" and args.flow_boundary_objective != "kde_wgf":
        raise ValueError("--flow-boundary-objective only applies to --method flow")
    uses_official_gmd = (
        args.method == "drift" and args.drift_objective == "official_gmd"
    ) or (
        args.method == "flow" and args.flow_boundary_objective == "official_gmd"
    )
    if args.method == "flow" and args.flow_boundary_objective == "official_gmd":
        if args.lambda_feature_boundary <= 0:
            raise ValueError(
                "--lambda-feature-boundary must be positive for an official_gmd "
                "flow boundary"
            )
        if args.lambda_boundary_origin:
            raise ValueError(
                "the official_gmd flow boundary replaces the raw-KDE origin boundary; "
                "pass --lambda-boundary-origin 0"
            )
    if uses_official_gmd:
        if args.drift_features == "mae" and args.feature_checkpoint is None:
            raise ValueError("--gmd-features mae requires --feature-checkpoint")
        if args.drift_features == "mae" and args.latent_normalization != "none":
            raise ValueError(
                "the CelebA MAE checkpoint is trained in raw latent coordinates; "
                "use --latent-normalization none"
            )
        if args.feature_checkpoint is not None and not args.feature_checkpoint.is_file():
            raise FileNotFoundError(
                f"feature checkpoint not found: {args.feature_checkpoint}"
            )
    for name in ("num_workers", "log_every", "checkpoint_every", "sample_every"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.log_every == 0:
        raise ValueError("--log-every must be positive")
    if args.sample_count < 1 or args.diagnostic_count < 2 or args.decode_batch < 1:
        raise ValueError("sample/decode counts must be positive and diagnostic count at least 2")
    if args.sample_time_power <= 0:
        raise ValueError("--sample-time-power must be positive")
    if args.train_start_index < 0:
        raise ValueError("--train-start-index must be non-negative")
    if args.train_end_index is not None and args.train_end_index <= args.train_start_index:
        raise ValueError("--train-end-index must exceed --train-start-index")
    if args.latent_prior == "train-full-gaussian" and args.latent_normalization != "none":
        raise ValueError(
            "--latent-prior train-full-gaussian is defined in raw latent coordinates; "
            "use --latent-normalization none"
        )
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
    if args.initial_model_checkpoint is not None:
        if args.resume is not None:
            raise ValueError("--initial-model-checkpoint cannot be combined with --resume")
        if not args.initial_model_checkpoint.is_file():
            raise FileNotFoundError(
                f"initial model checkpoint not found: {args.initial_model_checkpoint}"
            )
    if args.endpoint_teacher_checkpoint is not None:
        if args.method != "flow":
            raise ValueError("--endpoint-teacher-checkpoint only applies to --method flow")
        if not args.endpoint_teacher_checkpoint.is_file():
            raise FileNotFoundError(
                f"endpoint teacher checkpoint not found: {args.endpoint_teacher_checkpoint}"
            )
        if args.lambda_endpoint_teacher <= 0:
            raise ValueError(
                "--lambda-endpoint-teacher must be positive when an endpoint teacher is used"
            )
    elif args.lambda_endpoint_teacher:
        raise ValueError(
            "--lambda-endpoint-teacher requires --endpoint-teacher-checkpoint"
        )
    if not args.cache.is_file():
        raise FileNotFoundError(f"latent cache not found: {args.cache}")
    if args.vae_scaling_factor is not None and args.vae_scaling_factor <= 0:
        raise ValueError("--vae-scaling-factor must be positive")
    if args.decode_samples and not args.vae_model.is_dir():
        raise FileNotFoundError(f"VAE directory not found: {args.vae_model}")
    if args.decode_samples and importlib.util.find_spec("diffusers") is None:
        raise ImportError(
            "--decode-samples requires diffusers; use the ROCm environment that built the cache "
            "or pass --no-decode-samples"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.command = shlex.join([sys.argv[0], *(sys.argv[1:] if argv is None else argv)])
    apply_smoke_overrides(args)
    args.rbf_geometry = (
        "dimension_normalized" if args.rbf_geometry == "normalized" else args.rbf_geometry
    )
    validate_args(args)
    return args


def setup_distributed(args: argparse.Namespace) -> None:
    args.world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.rank = int(os.environ.get("RANK", "0"))
    args.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    args.distributed = args.world_size > 1
    if args.smoke and args.distributed:
        raise ValueError("--smoke is a single-process CPU preflight")

    requested = args.device
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    requested_device = torch.device(requested)
    if args.distributed:
        if requested_device.type != "cuda":
            raise ValueError("multi-process training requires a CUDA/ROCm device")
        if args.distributed_backend != "nccl":
            raise ValueError("GPU torchrun must use --distributed-backend nccl (NCCL/RCCL)")
        if not torch.cuda.is_available():
            raise RuntimeError("torchrun requested GPU DDP but torch.cuda is unavailable")
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
    else:
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA/ROCm device requested but torch.cuda is unavailable")
        args.device = requested_device
        if args.device.type == "cuda" and args.device.index is not None:
            torch.cuda.set_device(args.device)
    args.global_batch = args.batch * args.world_size

    if args.distributed and args.global_kde_support and args.time_groups > 1 and not args.synchronize_times:
        raise ValueError(
            "global grouped KDE requires synchronized times so each gathered group represents one time"
        )


def cleanup_distributed(args: argparse.Namespace) -> None:
    if getattr(args, "distributed", False) and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args: argparse.Namespace) -> bool:
    return args.rank == 0


def barrier(args: argparse.Namespace) -> None:
    if args.distributed:
        dist.barrier()


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def run_on_rank_zero(args: argparse.Namespace, operation: Callable[[], Any]) -> Any:
    """Run filesystem/artifact work on rank zero and propagate failures to all ranks."""

    barrier(args)
    result = None
    failure = None
    if is_main_process(args):
        try:
            result = operation()
        except Exception:
            failure = traceback.format_exc()
    if args.distributed:
        payload = [failure]
        dist.broadcast_object_list(payload, src=0)
        failure = payload[0]
    if failure is not None:
        raise RuntimeError(f"rank-zero operation failed:\n{failure}")
    barrier(args)
    return result


def seed_process(args: argparse.Namespace) -> None:
    # All ranks construct identical parameters; stochastic training is then
    # decorrelated by rank after DDP's initial parameter broadcast.
    torch.manual_seed(args.seed)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def resolve_run_directory(args: argparse.Namespace) -> Path:
    path: Path | None = None
    if is_main_process(args):
        if args.run_dir is not None:
            path = args.run_dir.resolve()
        elif args.resume is not None:
            checkpoint_parent = args.resume.resolve().parent
            path = checkpoint_parent.parent if checkpoint_parent.name == "checkpoints" else checkpoint_parent
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            safe_label = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in args.run_label
            ).strip("_")
            run_name = f"{stamp}_{safe_label}" if safe_label else stamp
            path = (args.output_root / args.method / run_name).resolve()
        path.mkdir(parents=True, exist_ok=True)
    if args.distributed:
        payload = [str(path) if path is not None else ""]
        dist.broadcast_object_list(payload, src=0)
        path = Path(payload[0])
    assert path is not None
    return path


def select_training_latents(
    latents: torch.Tensor,
    *,
    start_index: int,
    end_index: int | None,
    smoke: bool,
    smoke_count: int,
) -> tuple[torch.Tensor, int, int]:
    end = latents.shape[0] if end_index is None else end_index
    if smoke:
        end = min(end, start_index + smoke_count)
    if start_index >= latents.shape[0] or end > latents.shape[0]:
        raise ValueError(
            f"requested train indices [{start_index}, {end}) outside cache of {latents.shape[0]} items"
        )
    selected = flatten_latents(latents[start_index:end])
    return selected, start_index, end


def prepare_training_latents(
    raw_training_latents: torch.Tensor,
    normalization: LatentAffineNormalization,
) -> torch.Tensor:
    """Apply the fitted coordinates and use the model's canonical float32 dtype."""

    # Cast before the affine transform.  Otherwise a supported float16 cache
    # silently rounds the fitted channel mean/std and performs the coordinate
    # transform in half precision before the final cast.
    return normalization.normalize(raw_training_latents.to(dtype=torch.float32))


def fit_training_prior(
    args: argparse.Namespace,
    raw_training_latents: torch.Tensor,
    train_start: int,
    train_end: int,
) -> LatentGaussianPrior:
    """Resolve the source prior identically on every rank.

    The dense train-only fit and cache hashing happen once on rank zero.  Its
    fully self-validating CPU payload is then broadcast to the other ranks;
    fitting itself consumes no random numbers.
    """

    mode = getattr(args, "latent_prior", "standard")
    if mode == "standard":
        return LatentGaussianPrior.standard()
    if mode != "train-full-gaussian":
        raise ValueError(f"unsupported latent prior: {mode!r}")
    payload: list[dict[str, Any] | None] = [None]
    failure: str | None = None
    if is_main_process(args):
        try:
            if getattr(args, "resume", None) is not None:
                checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
                if (
                    not isinstance(checkpoint, Mapping)
                    or checkpoint.get("format") != CHECKPOINT_FORMAT
                    or checkpoint.get("version") != CHECKPOINT_VERSION
                    or not isinstance(checkpoint.get("latent_prior"), Mapping)
                ):
                    raise ValueError(
                        "full-Gaussian resume requires a compatible checkpoint with embedded prior state"
                    )
                fitted = LatentGaussianPrior.from_payload(checkpoint["latent_prior"])
                expected_cache_sha256 = sha256_file(args.cache)
                expected_values_sha256 = sha256_tensor(
                    unflatten_latents(
                        raw_training_latents.detach().to(device="cpu", dtype=torch.float64)
                    )
                )
                if (
                    fitted.mode != "train_full_gaussian"
                    or fitted.train_start_index != train_start
                    or fitted.train_end_index != train_end
                    or fitted.sample_count != raw_training_latents.shape[0]
                    or fitted.source_cache_sha256 != expected_cache_sha256
                    or fitted.source_values_sha256 != expected_values_sha256
                ):
                    raise ValueError(
                        "resume checkpoint latent prior disagrees with the current cache or training split"
                    )
            else:
                fitted = LatentGaussianPrior.fit_full(
                    raw_training_latents,
                    train_start_index=train_start,
                    train_end_index=train_end,
                    source_cache_sha256=sha256_file(args.cache),
                )
            payload[0] = fitted.to_payload()
        except Exception:
            failure = traceback.format_exc()
    if args.distributed:
        failures = [failure]
        dist.broadcast_object_list(failures, src=0)
        if failures[0] is not None:
            raise RuntimeError(f"rank-zero latent-prior fit failed:\n{failures[0]}")
        dist.broadcast_object_list(payload, src=0)
    elif failure is not None:
        raise RuntimeError(f"latent-prior fit failed:\n{failure}")
    if payload[0] is None:
        raise RuntimeError("rank zero did not provide a fitted latent prior")
    return LatentGaussianPrior.from_payload(payload[0])


def ensure_latent_prior_sidecar(run_dir: Path, prior: LatentGaussianPrior) -> Path | None:
    """Atomically create, or strictly validate, a nonstandard-prior sidecar."""

    if prior.mode == "standard":
        return None
    path = run_dir / LATENT_PRIOR_FILENAME
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        found = LatentGaussianPrior.from_payload(payload)
        if not found.same_definition(prior):
            raise ValueError(f"existing latent-prior sidecar disagrees with this run: {path}")
    else:
        atomic_torch_save(path, prior.to_payload())
    return path


class LatentBatchStream:
    """Deterministic epoch sampler whose exact position is checkpointable."""

    def __init__(self, latents: torch.Tensor, args: argparse.Namespace) -> None:
        dataset = TensorDataset(latents)
        self.sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        loader_generator = torch.Generator().manual_seed(args.seed + 10_000 + args.rank)
        self.loader = DataLoader(
            dataset,
            batch_size=args.batch,
            sampler=self.sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory and args.device.type == "cuda",
            drop_last=True,
            persistent_workers=args.num_workers > 0,
            generator=loader_generator,
        )
        if len(self.loader) == 0:
            raise ValueError(
                "training split is too small for batch/world size after distributed drop_last"
            )
        self.epoch = 0
        self.batch_offset = 0
        self._iterator = None
        self._reset_iterator()

    def _reset_iterator(self) -> None:
        self.sampler.set_epoch(self.epoch)
        self._iterator = iter(self.loader)

    def next(self) -> torch.Tensor:
        assert self._iterator is not None
        try:
            (batch,) = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.batch_offset = 0
            self._reset_iterator()
            assert self._iterator is not None
            (batch,) = next(self._iterator)
        self.batch_offset += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch, "batch_offset": self.batch_offset}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        epoch = int(state.get("epoch", 0))
        batch_offset = int(state.get("batch_offset", 0))
        if epoch < 0 or not 0 <= batch_offset <= len(self.loader):
            raise ValueError("invalid data-stream checkpoint state")
        self.epoch = epoch
        self.batch_offset = 0
        self._reset_iterator()
        assert self._iterator is not None
        for _ in range(batch_offset):
            try:
                next(self._iterator)
            except StopIteration as error:
                raise ValueError("data-stream checkpoint offset exceeds epoch") from error
            self.batch_offset += 1


class LatentEndpointModel(nn.Module):
    """Endpoint-only baseline using the same latent U-Net at Phi(., 0, 1)."""

    def __init__(self, flow_map: TransitionConditionedLatentUNet) -> None:
        super().__init__()
        self.flow_map = flow_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        start = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return self.flow_map(x, start, torch.ones_like(start))


def make_model(args: argparse.Namespace) -> nn.Module:
    flow_map = TransitionConditionedLatentUNet(
        base_channels=args.base_channels,
        time_embedding_dim=args.time_embedding_dim,
        condition_dim=args.condition_dim,
        zero_init_endpoint=args.zero_init_endpoint,
        endpoint_parameterization=getattr(args, "endpoint_parameterization", "absolute"),
    )
    return flow_map if args.method == "flow" else LatentEndpointModel(flow_map)


def load_initial_model_weights(
    model: nn.Module,
    checkpoint: str | Path,
    *,
    weight_kind: str = "ema",
) -> str:
    """Load only model weights, allowing endpoint-wrapper/flow-map conversion.

    Endpoint GMD runs wrap the transition network under ``flow_map`` whereas
    flow-map runs store that same network directly.  This loader accepts either
    exported state dicts or full training checkpoints, strips only the known
    ``module``/``flow_map`` wrappers, and still requires an exact key-and-shape
    match.  Optimizer, EMA counters, data-stream state and RNG are deliberately
    not imported: this is a new training stage, not a resume operation.
    """

    if weight_kind not in {"raw", "ema"}:
        raise ValueError("weight_kind must be 'raw' or 'ema'")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise ValueError("initial model checkpoint must contain a mapping")

    if payload and all(torch.is_tensor(value) for value in payload.values()):
        state = dict(payload)
        source = "exported_state_dict"
    else:
        source_key = "model" if weight_kind == "raw" else "ema"
        state_payload = payload.get(source_key)
        if not isinstance(state_payload, Mapping):
            raise ValueError(
                f"training checkpoint does not contain a {weight_kind} model state"
            )
        state = {
            str(key): value
            for key, value in state_payload.items()
            if key != "n_averaged" and torch.is_tensor(value)
        }
        source = f"training_checkpoint_{weight_kind}"
    if not state:
        raise ValueError("initial model state is empty")

    target = model.state_dict()

    def strip_prefix(values: Mapping[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
        if not values or not all(key.startswith(prefix) for key in values):
            return dict(values)
        return {key[len(prefix) :]: value for key, value in values.items()}

    candidates: list[tuple[str, dict[str, torch.Tensor]]] = [("identity", state)]
    without_module = strip_prefix(state, "module.")
    candidates.append(("strip_module", without_module))
    candidates.append(("strip_flow_map", strip_prefix(state, "flow_map.")))
    candidates.append(
        ("strip_module_flow_map", strip_prefix(without_module, "flow_map."))
    )
    candidates.append(
        ("add_flow_map", {f"flow_map.{key}": value for key, value in without_module.items()})
    )

    for transform, candidate in candidates:
        if candidate.keys() != target.keys():
            continue
        if any(candidate[key].shape != target[key].shape for key in target):
            continue
        model.load_state_dict(candidate, strict=True)
        return f"{source}:{transform}"

    missing = sorted(set(target).difference(state))[:5]
    unexpected = sorted(set(state).difference(target))[:5]
    raise ValueError(
        "initial model weights do not exactly match the requested architecture; "
        f"missing examples={missing}, unexpected examples={unexpected}"
    )


def make_endpoint_teacher(args: argparse.Namespace) -> nn.Module | None:
    """Build an immutable endpoint teacher for the second flow-map stage."""

    checkpoint = getattr(args, "endpoint_teacher_checkpoint", None)
    if checkpoint is None:
        return None
    teacher = make_model(args).to(args.device)
    args._endpoint_teacher_load_transform = load_initial_model_weights(
        teacher,
        checkpoint,
        weight_kind=args.endpoint_teacher_weight_kind,
    )
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def make_drifting_feature_encoder(
    args: argparse.Namespace,
    cache_metadata: Mapping[str, Any] | None = None,
) -> nn.Module | None:
    """Load the frozen learned geometry used by the released image recipe."""

    direct_gmd = (
        args.method == "drift"
        and getattr(args, "drift_objective", "kde_wgf") == "official_gmd"
    )
    flow_boundary_gmd = (
        args.method == "flow"
        and getattr(args, "flow_boundary_objective", "kde_wgf") == "official_gmd"
    )
    if not (direct_gmd or flow_boundary_gmd) or getattr(
        args, "drift_features", "raw_pyramid"
    ) == "raw_pyramid":
        return None
    payload = torch.load(args.feature_checkpoint, map_location="cpu", weights_only=True)
    feature_metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
    if not isinstance(feature_metadata, Mapping):
        raise ValueError("feature checkpoint lacks training metadata")
    expected_feature_metadata = {
        "cache_sha256": sha256_file(args.cache),
        "cache_model_id": (cache_metadata or {}).get("model_id"),
        "cache_encoding": (cache_metadata or {}).get("encoding"),
        "cache_scaling_factor": (cache_metadata or {}).get("scaling_factor"),
        "train_start_index": args.train_start_index,
        "train_end_index": args.train_end_index,
        "latent_normalization": "none",
    }
    mismatches = [
        key
        for key, expected in expected_feature_metadata.items()
        if feature_metadata.get(key) != expected
    ]
    if mismatches:
        raise ValueError(
            "feature checkpoint disagrees with the generator data coordinates for "
            f"{mismatches}"
        )
    from src.celeba_gmd_features import CHECKPOINT_FORMAT as FEATURE_V1_FORMAT
    from src.celeba_gmd_features import load_frozen_feature_encoder
    from src.celeba_gmd_features_v2 import CHECKPOINT_FORMAT as FEATURE_V2_FORMAT
    from src.celeba_gmd_features_v2 import load_frozen_feature_encoder_v2

    feature_format = payload.get("format") if isinstance(payload, Mapping) else None
    if feature_format == FEATURE_V1_FORMAT:
        encoder = load_frozen_feature_encoder(payload, args.device)
    elif feature_format == FEATURE_V2_FORMAT:
        encoder = load_frozen_feature_encoder_v2(payload, args.device)
    else:
        raise ValueError(
            "unsupported feature checkpoint format; expected a versioned CelebA "
            f"MAE v1 or v2 checkpoint, observed {feature_format!r}"
        )
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def make_ema(model: nn.Module, args: argparse.Namespace) -> AveragedModel | None:
    if args.ema_decay == 0:
        return None
    ema = AveragedModel(
        unwrap_model(model),
        multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
        use_buffers=True,
    )
    ema.requires_grad_(False)
    ema.eval()
    if getattr(args, "ema_warmup_steps", 1) == 0:
        # AveragedModel would otherwise copy theta_1 on its first update.
        # Mark theta_0 as the first observation so multi_avg_fn performs the
        # released recursion: ema_1 = decay * theta_0 + (1-decay) * theta_1.
        ema.n_averaged.fill_(1)
    return ema


def update_ema(
    ema: AveragedModel | None,
    raw_model: nn.Module,
    *,
    completed_step: int,
    warmup_steps: int,
    started: bool,
) -> bool:
    if ema is None or completed_step < warmup_steps:
        return started
    if not started:
        ema.module.load_state_dict(raw_model.state_dict())
        ema.n_averaged.zero_()
    ema.update_parameters(raw_model)
    return True


def reduce_stats(stats: dict[str, Any], args: argparse.Namespace) -> dict[str, torch.Tensor]:
    keys: list[str] = []
    tensors: list[torch.Tensor] = []
    for key, value in stats.items():
        tensor = (
            value.detach().to(device=args.device, dtype=torch.float32)
            if torch.is_tensor(value)
            else torch.tensor(float(value), device=args.device)
        )
        if tensor.numel() != 1:
            continue
        keys.append(key)
        tensors.append(tensor.reshape(()))
    if not tensors:
        return {}
    # A GMD step exposes roughly one hundred scalar diagnostics.  Reducing
    # them one by one adds network latency without changing their semantics;
    # one packed collective is exactly the same arithmetic mean.
    packed = torch.stack(tensors)
    if args.distributed:
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.div_(args.world_size)
    return {key: packed[index] for index, key in enumerate(keys)}


def gather_official_feature_support(
    generated_features: Mapping[str, torch.Tensor],
    positive_features: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """Gather unconditional GMD targets while keeping queries rank-local.

    The released conditional recipe compares 64 generated particles with 64
    positives *inside each class*.  Unconditional CelebA places all face modes
    in one group, so this optional ablation exposes each local query to the
    global DDP support.  Generated supports are detached by the drifting loss;
    only the local query tensor therefore needs an autograd graph.

    Feature families are packed into one matrix per sample, reducing this to
    two collectives per update regardless of how many feature families exist.
    """

    if generated_features.keys() != positive_features.keys():
        raise ValueError("generated and positive feature dictionaries must have identical keys")
    if not (
        getattr(args, "distributed", False)
        and getattr(args, "drift_global_particle_support", False)
    ):
        return dict(positive_features), None
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("global drifting particle support requires an initialized process group")

    names = tuple(generated_features)
    if not names:
        raise ValueError("at least one feature family is required")
    first = generated_features[names[0]]
    if first.ndim != 3:
        raise ValueError("feature tensors must have shape [samples, tokens, dimensions]")
    local_count = first.shape[0]
    device, dtype = first.device, first.dtype
    shapes: dict[str, tuple[int, int]] = {}
    for name in names:
        generated = generated_features[name]
        positive = positive_features[name]
        if generated.ndim != 3 or positive.ndim != 3:
            raise ValueError("feature tensors must have shape [samples, tokens, dimensions]")
        if generated.shape != positive.shape:
            raise ValueError(
                "global particle support requires equal local generated/positive shapes"
            )
        if generated.shape[0] != local_count:
            raise ValueError("all feature families must share their local sample count")
        if generated.device != device or positive.device != device:
            raise ValueError("all feature families must share one device")
        if generated.dtype != dtype or positive.dtype != dtype:
            raise ValueError("all feature families must share one dtype")
        shapes[name] = (generated.shape[1], generated.shape[2])

    def pack(features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [features[name].detach().reshape(local_count, -1) for name in names],
            dim=1,
        ).contiguous()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    def gather(local: torch.Tensor) -> torch.Tensor:
        output = torch.empty(
            (world_size * local_count, local.shape[1]),
            device=device,
            dtype=dtype,
        )
        dist.all_gather_into_tensor(output, local)
        return output

    gathered_generated = gather(pack(generated_features))
    gathered_positive = gather(pack(positive_features))
    remote_generated = torch.cat(
        (
            gathered_generated[: rank * local_count],
            gathered_generated[(rank + 1) * local_count :],
        ),
        dim=0,
    )

    def unpack(packed: torch.Tensor) -> dict[str, torch.Tensor]:
        unpacked: dict[str, torch.Tensor] = {}
        offset = 0
        for name in names:
            tokens, dimensions = shapes[name]
            width = tokens * dimensions
            unpacked[name] = packed[:, offset : offset + width].reshape(
                packed.shape[0], tokens, dimensions
            )
            offset += width
        if offset != packed.shape[1]:
            raise RuntimeError("packed drifting feature width is inconsistent")
        return unpacked

    return unpack(gathered_positive), unpack(remote_generated)


def distillation_span(args: argparse.Namespace, step_index: int) -> float:
    progress = step_index / max(args.steps - 1, 1)
    return args.distill_min_span + (args.distill_max_span - args.distill_min_span) * progress


def official_gmd_sample_loss(
    generated: torch.Tensor,
    data: torch.Tensor,
    args: argparse.Namespace,
    feature_encoder: nn.Module | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the released GMD loss to generated and positive latent samples.

    This helper is shared by the endpoint-only baseline and by the proposed
    two-time flow map.  The positive branch and every particle target remain
    stop-gradient exactly as in the released loss, while the frozen feature
    encoder remains differentiable with respect to ``generated``.
    """

    if getattr(args, "drift_features", "raw_pyramid") == "raw_pyramid":
        generated_features = raw_latent_pyramid_features(generated)
        with torch.no_grad():
            positive_features = raw_latent_pyramid_features(data)
    else:
        if feature_encoder is None:
            raise RuntimeError("official MAE drifting requires a frozen feature encoder")
        generated_features = feature_encoder.encode_features(
            unflatten_latents(generated)
        )
        with torch.no_grad():
            positive_features = feature_encoder.encode_features(
                unflatten_latents(data)
            )
    positive_features, fixed_negative_features = gather_official_feature_support(
        generated_features,
        positive_features,
        args,
    )
    loss, statistics = summed_feature_drifting_loss(
        generated_features,
        positive_features,
        fixed_negative_features=fixed_negative_features,
        temperatures=getattr(args, "drift_temperatures", OFFICIAL_TEMPERATURES),
        distributed_statistics=(
            getattr(args, "distributed", False)
            and getattr(args, "drift_global_statistics", True)
        ),
    )
    statistics.update(
        {
            "generated_mean": generated.detach().mean(),
            "generated_std": generated.detach().std(),
            "generated_rms": generated.detach().square().mean().sqrt(),
            "feature_family_count": float(len(generated_features)),
            "positive_support_count": float(
                next(iter(positive_features.values())).shape[0]
            ),
            "generated_support_count": float(
                generated.shape[0]
                + (
                    0
                    if fixed_negative_features is None
                    else next(iter(fixed_negative_features.values())).shape[0]
                )
            ),
        }
    )
    return loss, statistics


def training_loss(
    model: nn.Module,
    data: torch.Tensor,
    args: argparse.Namespace,
    step_index: int,
    prior_sampler: LatentGaussianSampler | None = None,
    feature_encoder: nn.Module | None = None,
    endpoint_teacher: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    z = torch.randn(data.shape[0], CELEBA_LATENT_DIM, device=args.device, dtype=data.dtype)
    z_ref = torch.randn_like(z)
    if prior_sampler is not None:
        z = prior_sampler.transform_standard(z)
        z_ref = prior_sampler.transform_standard(z_ref)
    if (
        args.method == "drift"
        and getattr(args, "drift_objective", "kde_wgf") == "official_gmd"
    ):
        generated = model(z)
        return official_gmd_sample_loss(
            generated,
            data,
            args,
            feature_encoder,
        )
    if args.method == "drift":
        return optimizer_time_drifting_loss(
            model,
            z,
            z_ref,
            data,
            step_size=args.drift_step_size,
            velocity_kind=args.velocity,
            bandwidth=args.bandwidth,
            rbf_geometry=args.rbf_geometry,
            global_kde_support=args.global_kde_support,
        )
    feature_boundary = (
        getattr(args, "flow_boundary_objective", "kde_wgf") == "official_gmd"
    )
    flow_loss, statistics = fast_wgf_flow_map_loss(
        model,
        z,
        z_ref,
        data,
        dt=args.dt,
        schedule=args.schedule,
        time_scale=args.time_scale,
        eps=args.schedule_eps,
        path_weight=args.path_weight,
        lambda_wgf=args.lambda_wgf,
        lambda_lagrangian=args.lambda_lagrangian,
        lambda_eulerian=args.lambda_eulerian,
        lambda_sg=args.lambda_semigroup,
        lambda_terminal_sg=getattr(args, "lambda_terminal_semigroup", 0.0),
        lambda_boundary=0.0,
        lambda_boundary_origin=(0.0 if feature_boundary else args.lambda_boundary_origin),
        lambda_boundary_arbitrary=args.lambda_boundary_arbitrary,
        boundary_step_size=args.boundary_step_size,
        distill_max_span=distillation_span(args, step_index),
        time_groups=args.time_groups,
        velocity_kind=args.velocity,
        bandwidth=args.bandwidth,
        rbf_geometry=args.rbf_geometry,
        global_kde_support=args.global_kde_support,
        synchronize_times=args.synchronize_times,
    )
    if not feature_boundary and endpoint_teacher is None:
        return flow_loss, statistics

    endpoint_time = torch.ones(data.shape[0], device=args.device, dtype=data.dtype)
    endpoint = model(z, torch.zeros_like(endpoint_time), endpoint_time)
    total = flow_loss
    if feature_boundary:
        endpoint_loss, endpoint_statistics = official_gmd_sample_loss(
            endpoint,
            data,
            args,
            feature_encoder,
        )
        feature_weight = getattr(args, "lambda_feature_boundary", 0.003)
        total = total + feature_weight * endpoint_loss
        statistics.update(
            {
                f"gmd_boundary/{key}": value
                for key, value in endpoint_statistics.items()
                if key != "loss"
            }
        )
        statistics.update(
            {
                "loss_feature_boundary": endpoint_loss.detach(),
                "lambda_feature_boundary": feature_weight,
            }
        )
    if endpoint_teacher is not None:
        with torch.no_grad():
            teacher_endpoint = endpoint_teacher(
                z,
                torch.zeros_like(endpoint_time),
                endpoint_time,
            )
        endpoint_teacher_loss = F.mse_loss(endpoint, teacher_endpoint)
        teacher_weight = getattr(args, "lambda_endpoint_teacher", 0.0)
        total = total + teacher_weight * endpoint_teacher_loss
        statistics.update(
            {
                "loss_endpoint_teacher": endpoint_teacher_loss.detach(),
                "lambda_endpoint_teacher": teacher_weight,
                "endpoint_teacher_rmse": endpoint_teacher_loss.detach().sqrt(),
            }
        )
    statistics.update(
        {
            "loss": total.detach(),
            "loss_flow_without_feature_boundary": flow_loss.detach(),
        }
    )
    return total, statistics


def weighted_loss_stats(stats: dict[str, torch.Tensor], args: argparse.Namespace) -> dict[str, float]:
    if args.method != "flow":
        return {}
    values = {key: float(value.detach().cpu()) for key, value in stats.items()}
    return {
        "weighted_wgf": args.path_weight * args.lambda_wgf * values.get("loss_wgf", 0.0),
        "weighted_eulerian": args.path_weight
        * args.lambda_eulerian
        * values.get("loss_eulerian", 0.0),
        "weighted_lagrangian": args.path_weight
        * args.lambda_lagrangian
        * values.get("loss_lagrangian", 0.0),
        "weighted_semigroup": args.lambda_semigroup * values.get("loss_sg", 0.0),
        "weighted_terminal_semigroup": getattr(args, "lambda_terminal_semigroup", 0.0)
        * values.get("loss_terminal_semigroup", 0.0),
        "weighted_boundary_origin": args.lambda_boundary_origin
        * values.get("loss_boundary_origin", 0.0),
        "weighted_boundary_arbitrary": args.lambda_boundary_arbitrary
        * values.get("loss_boundary_arbitrary", 0.0),
        "weighted_feature_boundary": getattr(args, "lambda_feature_boundary", 0.0)
        * values.get("loss_feature_boundary", 0.0),
        "weighted_endpoint_teacher": getattr(args, "lambda_endpoint_teacher", 0.0)
        * values.get("loss_endpoint_teacher", 0.0),
    }


def training_signature(
    args: argparse.Namespace,
    cache_metadata: dict[str, Any],
    train_start: int,
    train_end: int,
    normalization: LatentAffineNormalization | None = None,
    prior: LatentGaussianPrior | None = None,
    *,
    cache_sha256: str | None = None,
    train_values_sha256: str | None = None,
) -> dict[str, Any]:
    names = (
        "method",
        "batch",
        "base_channels",
        "time_embedding_dim",
        "condition_dim",
        "zero_init_endpoint",
        "deterministic",
        "seed",
        "velocity",
        "bandwidth",
        "rbf_geometry",
        "global_kde_support",
        "synchronize_times",
        "time_groups",
        "dt",
        "schedule",
        "time_scale",
        "schedule_eps",
        "path_weight",
        "lambda_wgf",
        "lambda_eulerian",
        "lambda_lagrangian",
        "lambda_boundary_origin",
        "lambda_boundary_arbitrary",
        "boundary_step_size",
        "distill_min_span",
        "distill_max_span",
        "drift_step_size",
        "lr",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "ema_decay",
        "ema_warmup_steps",
        "grad_clip",
        "world_size",
    )
    signature = {
        **{name: getattr(args, name) for name in names},
        "signature_version": TRAINING_SIGNATURE_VERSION,
        "latent_dim": CELEBA_LATENT_DIM,
        "prior": "N(0,I)",
        "cache_model_id": cache_metadata.get("model_id"),
        "cache_scaling_factor": cache_metadata.get("scaling_factor"),
        "cache_encoding": cache_metadata.get("encoding"),
        "train_start_index": train_start,
        "train_end_index": train_end,
    }
    if getattr(args, "lr_warmup_steps", 0):
        signature["lr_warmup_steps"] = args.lr_warmup_steps
    if getattr(args, "drift_objective", "kde_wgf") != "kde_wgf":
        signature.update(
            {
                "drift_objective": args.drift_objective,
                "drift_features": args.drift_features,
                "drift_temperatures": list(args.drift_temperatures),
                "drift_global_statistics": args.drift_global_statistics,
            }
        )
        # Preserve resume compatibility with official-GMD checkpoints created
        # before global particle support was exposed.  The enabled mode changes
        # the mathematical target support and must remain part of the signature;
        # absence denotes the historical/default local-support mode.
        if getattr(args, "drift_global_particle_support", False):
            signature["drift_global_particle_support"] = True
    if getattr(args, "flow_boundary_objective", "kde_wgf") != "kde_wgf":
        signature.update(
            {
                "flow_boundary_objective": args.flow_boundary_objective,
                "lambda_feature_boundary": args.lambda_feature_boundary,
                "gmd_features": args.drift_features,
                "gmd_temperatures": list(args.drift_temperatures),
                "gmd_global_statistics": args.drift_global_statistics,
                "gmd_global_particle_support": args.drift_global_particle_support,
            }
        )
    # Preserve strict resume compatibility with checkpoints produced before
    # the optional semigroup anchor was exposed by the CLI.  Non-zero values
    # remain part of the mathematical training signature.
    if args.lambda_semigroup:
        signature["lambda_semigroup"] = args.lambda_semigroup
    if getattr(args, "lambda_terminal_semigroup", 0.0):
        signature["lambda_terminal_semigroup"] = args.lambda_terminal_semigroup
    # Keep mode=none byte-for-byte compatible with checkpoints created before
    # normalization existed.  A fitted transform, by contrast, is part of the
    # mathematical training coordinates and must match exactly on resume.
    if normalization is not None and normalization.mode != "none":
        signature["latent_normalization"] = normalization.metadata(
            train_start_index=train_start,
            train_end_index=train_end,
        )
    prior = prior or LatentGaussianPrior.standard()
    if prior.mode != "standard":
        signature["prior"] = prior.mode
        signature["latent_prior"] = prior.metadata()
    # Absence means the historical absolute endpoint, preserving old resume
    # signatures exactly. Residual endpoints are a structural model choice.
    if getattr(args, "endpoint_parameterization", "absolute") != "absolute":
        signature["endpoint_parameterization"] = args.endpoint_parameterization
    if cache_sha256 is not None:
        signature["cache_sha256"] = cache_sha256
    if train_values_sha256 is not None:
        signature["train_values_sha256"] = train_values_sha256
    if getattr(args, "feature_checkpoint", None) is not None:
        signature["feature_checkpoint_sha256"] = sha256_file(
            args.feature_checkpoint
        )
    if getattr(args, "initial_model_checkpoint", None) is not None:
        signature.update(
            {
                "initial_model_checkpoint_sha256": sha256_file(
                    args.initial_model_checkpoint
                ),
                "initial_weight_kind": args.initial_weight_kind,
            }
        )
    if getattr(args, "endpoint_teacher_checkpoint", None) is not None:
        signature.update(
            {
                "endpoint_teacher_checkpoint_sha256": sha256_file(
                    args.endpoint_teacher_checkpoint
                ),
                "endpoint_teacher_weight_kind": args.endpoint_teacher_weight_kind,
                "lambda_endpoint_teacher": args.lambda_endpoint_teacher,
            }
        )
    return signature


def local_rng_state(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    state = {"cpu": torch.get_rng_state().cpu()}
    if args.device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(args.device).cpu()
    return state


def gather_rng_states(args: argparse.Namespace) -> list[dict[str, torch.Tensor]]:
    local = local_rng_state(args)
    if not args.distributed:
        return [local]
    gathered: list[dict[str, torch.Tensor] | None] = [None] * args.world_size
    dist.all_gather_object(gathered, local)
    return [state for state in gathered if state is not None]


def restore_rng_state(states: list[dict[str, torch.Tensor]], args: argparse.Namespace) -> None:
    if len(states) != args.world_size:
        raise ValueError(
            f"checkpoint has RNG state for {len(states)} ranks but current world size is {args.world_size}"
        )
    state = states[args.rank]
    torch.set_rng_state(state["cpu"].cpu())
    if args.device.type == "cuda":
        if "cuda" not in state:
            raise ValueError("checkpoint lacks CUDA/ROCm RNG state")
        torch.cuda.set_rng_state(state["cuda"].cpu(), args.device)


def save_training_checkpoint(
    path: str | Path,
    *,
    completed_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: AveragedModel | None,
    ema_started: bool,
    stream: LatentBatchStream,
    signature: dict[str, Any],
    normalization: LatentAffineNormalization | None = None,
    prior: LatentGaussianPrior | None = None,
    run_dir: Path,
    args: argparse.Namespace,
) -> None:
    rng_states = gather_rng_states(args)
    def write_checkpoint() -> None:
        payload = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "completed_step": completed_step,
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "ema_started": ema_started,
            "data_stream": stream.state_dict(),
            "rng_by_rank": rng_states,
            "training_signature": signature,
            "latent_normalization": (normalization or LatentAffineNormalization.identity()).metadata(),
            "run_dir": str(run_dir),
            "saved_at_utc": utc_now(),
        }
        if prior is not None and prior.mode != "standard":
            payload["latent_prior"] = prior.to_payload()
        atomic_torch_save(path, payload)

    run_on_rank_zero(args, write_checkpoint)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: AveragedModel | None,
    stream: LatentBatchStream,
    expected_signature: dict[str, Any],
    expected_prior: LatentGaussianPrior | None = None,
    args: argparse.Namespace,
) -> tuple[int, bool]:
    checkpoint = torch.load(Path(path), map_location=args.device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("--resume expects a full celeba_latent_flow checkpoint")
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {checkpoint.get('version')}")
    found_signature = checkpoint.get("training_signature")
    if not isinstance(found_signature, dict):
        raise ValueError("checkpoint lacks a valid training signature")
    expected_prior = expected_prior or LatentGaussianPrior.standard()
    structural_found = {
        "prior": found_signature.get("prior", "N(0,I)"),
        "endpoint_parameterization": found_signature.get(
            "endpoint_parameterization", "absolute"
        ),
    }
    structural_expected = {
        "prior": expected_signature.get("prior", "N(0,I)"),
        "endpoint_parameterization": expected_signature.get(
            "endpoint_parameterization", "absolute"
        ),
    }
    if structural_found != structural_expected:
        differing = [
            key for key in structural_expected if structural_found[key] != structural_expected[key]
        ]
        raise ValueError(
            "resume structural prior/endpoint signature differs for "
            f"{differing}; this cannot be overridden"
        )
    found_prior_metadata = found_signature.get("latent_prior")
    expected_prior_metadata = expected_signature.get("latent_prior")
    if expected_prior.mode == "standard":
        prior_metadata_match = found_prior_metadata is None and expected_prior_metadata is None
    else:
        prior_metadata_match = (
            isinstance(found_prior_metadata, Mapping)
            and isinstance(expected_prior_metadata, Mapping)
            and expected_prior.matches_metadata(found_prior_metadata)
            and expected_prior.matches_metadata(expected_prior_metadata)
        )
    if not prior_metadata_match:
        raise ValueError(
            "resume structural prior/endpoint signature differs for ['latent_prior']; "
            "this cannot be overridden"
        )
    immutable_data_mismatches = [
        key
        for key in ("cache_sha256", "train_values_sha256")
        if key in found_signature
        and key in expected_signature
        and found_signature[key] != expected_signature[key]
    ]
    if immutable_data_mismatches:
        raise ValueError(
            "resume data signature differs for "
            f"{immutable_data_mismatches}; this cannot be overridden"
        )
    found_signature_version = int(found_signature.get("signature_version", 1))
    if found_signature_version not in {1, TRAINING_SIGNATURE_VERSION}:
        raise ValueError(f"unsupported training signature version: {found_signature_version}")
    if found_signature_version == TRAINING_SIGNATURE_VERSION:
        invalid_attestations = [
            key
            for key in ("cache_sha256", "train_values_sha256")
            if not isinstance(found_signature.get(key), str)
            or len(found_signature[key]) != 64
            or any(character not in "0123456789abcdef" for character in found_signature[key])
        ]
        if invalid_attestations:
            raise ValueError(
                "version-2 checkpoint lacks valid immutable data attestations for "
                f"{invalid_attestations}"
            )
    legacy_unattested_keys: set[str] = set()
    if found_signature_version == 1:
        legacy_unattested_keys = {
            "signature_version",
            "seed",
            "grad_clip",
            "deterministic",
            "cache_sha256",
            "train_values_sha256",
        }
        legacy_run_dir = checkpoint.get("run_dir")
        legacy_config_path = (
            Path(legacy_run_dir) / "config.json" if isinstance(legacy_run_dir, str) else None
        )
        legacy_provenance_path = (
            Path(legacy_run_dir) / "provenance.json" if isinstance(legacy_run_dir, str) else None
        )
        checked_legacy_fields: list[str] = []
        if legacy_config_path is not None and legacy_config_path.is_file():
            legacy_config = json.loads(legacy_config_path.read_text(encoding="utf-8"))
            for key in ("seed", "grad_clip", "deterministic"):
                if key in legacy_config and key in expected_signature:
                    checked_legacy_fields.append(key)
                    if (
                        legacy_config[key] != expected_signature[key]
                        and not args.allow_resume_config_mismatch
                    ):
                        raise ValueError(
                            f"legacy resume config differs for ['{key}']; pass "
                            "--allow-resume-config-mismatch only intentionally"
                        )
        if legacy_provenance_path is not None and legacy_provenance_path.is_file():
            legacy_provenance = json.loads(legacy_provenance_path.read_text(encoding="utf-8"))
            saved_cache_sha256 = legacy_provenance.get("cache", {}).get("sha256")
            expected_cache_sha256 = expected_signature.get("cache_sha256")
            if saved_cache_sha256 is not None and expected_cache_sha256 is not None:
                checked_legacy_fields.append("cache_sha256")
                if saved_cache_sha256 != expected_cache_sha256:
                    raise ValueError(
                        "legacy resume cache SHA-256 differs; this cannot be overridden"
                    )
        missing_attestations = sorted(
            {"seed", "grad_clip", "deterministic", "cache_sha256"}
            - set(checked_legacy_fields)
        )
        if missing_attestations:
            warnings.warn(
                "legacy checkpoint signature cannot attest "
                f"{missing_attestations}; resume compatibility is limited",
                RuntimeWarning,
                stacklevel=2,
            )
    differing = sorted(
        key
        for key in set(found_signature).union(expected_signature)
        if key != "latent_prior"
        and key not in legacy_unattested_keys
        and found_signature.get(key) != expected_signature.get(key)
    )
    if differing and not args.allow_resume_config_mismatch:
        raise ValueError(
            "resume training signature differs for "
            f"{differing}; pass --allow-resume-config-mismatch only intentionally"
        )
    saved_prior_payload = checkpoint.get("latent_prior")
    if expected_prior.mode == "standard":
        if saved_prior_payload is not None:
            saved_prior = LatentGaussianPrior.from_payload(saved_prior_payload)
            if saved_prior.mode != "standard":
                raise ValueError("standard-prior resume checkpoint embeds a nonstandard prior")
    else:
        if not isinstance(saved_prior_payload, dict):
            raise ValueError("nonstandard-prior resume checkpoint lacks embedded prior state")
        saved_prior = LatentGaussianPrior.from_payload(saved_prior_payload)
        if not saved_prior.same_definition(expected_prior):
            raise ValueError("resume checkpoint latent-prior state differs from the train-only fit")
    unwrap_model(model).load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if ema is not None:
        if checkpoint.get("ema") is None:
            raise ValueError("current run enables EMA but checkpoint has no EMA state")
        ema.load_state_dict(checkpoint["ema"])
    elif checkpoint.get("ema") is not None and not args.allow_resume_config_mismatch:
        raise ValueError("checkpoint enables EMA but current run disables it")
    stream.load_state_dict(checkpoint["data_stream"])
    restore_rng_state(checkpoint["rng_by_rank"], args)
    return int(checkpoint["completed_step"]), bool(checkpoint.get("ema_started", False))


def _sample_model(
    model: nn.Module,
    method: str,
    z: torch.Tensor,
    nfe: int,
    time_power: float,
) -> torch.Tensor:
    if method == "flow":
        return sample_wgf_flow_map(model, z, nfe, time_power)
    x = z
    for _ in range(nfe):
        x = model(x)
    return x


def _safe_cosine(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(F.cosine_similarity(x, y, dim=1, eps=1e-8).mean().detach().cpu())


@torch.inference_mode()
def compute_transition_metrics(
    model: nn.Module,
    z: torch.Tensor,
    times: tuple[float, ...] = DEFAULT_TRANSITION_TIMES,
) -> list[dict[str, float]]:
    """Measure direct-vs-composed maps on canonical flow states."""

    zeros = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    canonical = [
        model(z, zeros, torch.full_like(zeros, end_time))
        for end_time in times
    ]
    rows: list[dict[str, float]] = []
    for start_index, start_time in enumerate(times):
        for end_index in range(start_index, len(times)):
            end_time = times[end_index]
            start_batch = torch.full_like(zeros, start_time)
            end_batch = torch.full_like(zeros, end_time)
            mapped = model(canonical[start_index], start_batch, end_batch)
            target = canonical[end_index]
            error = mapped - target
            rmse = error.square().mean().sqrt()
            target_delta = target - canonical[start_index]
            mapped_delta = mapped - canonical[start_index]
            target_rms = target_delta.square().mean().sqrt()
            rows.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "semigroup_rmse": float(rmse.detach().cpu()),
                    "relative_rmse": float((rmse / target_rms.clamp_min(1e-8)).detach().cpu()),
                    "delta_cosine": _safe_cosine(mapped_delta, target_delta),
                    "target_progress": float(
                        (
                            (mapped_delta * target_delta).sum(dim=1)
                            / target_delta.square().sum(dim=1).clamp_min(1e-8)
                        )
                        .mean()
                        .detach()
                        .cpu()
                    ),
                }
            )
    return rows


def _latent_sample_metrics(samples: dict[int, torch.Tensor]) -> list[dict[str, float]]:
    reference = samples[max(samples)]
    rows = []
    for nfe, sample in samples.items():
        difference = sample - reference
        rows.append(
            {
                "nfe": nfe,
                "latent_mean": float(sample.mean()),
                "latent_std": float(sample.std()),
                "latent_rms": float(sample.square().mean().sqrt()),
                "rmse_to_max_nfe": float(difference.square().mean().sqrt()),
                "cosine_to_max_nfe": _safe_cosine(sample, reference),
            }
        )
    return rows


def _mosaic(images: torch.Tensor) -> torch.Tensor:
    images = images.detach().cpu().clamp(0.0, 1.0)
    count = images.shape[0]
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / columns)
    _, channels, height, width = images.shape
    canvas = torch.zeros(channels, rows * height, columns * width)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[:, row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return canvas


def save_decoded_panel(
    decoded: dict[int, torch.Tensor],
    path: str | Path,
    *,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(decoded), figsize=(3.0 * len(decoded), 3.4))
    axes = list(axes.reshape(-1)) if hasattr(axes, "reshape") else [axes]
    for axis, (nfe, images) in zip(axes, decoded.items()):
        axis.imshow(_mosaic(images).permute(1, 2, 0))
        axis.set_title(f"NFE {nfe}")
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)


class ArtifactGenerator:
    def __init__(
        self,
        args: argparse.Namespace,
        run_dir: Path,
        cache_metadata: dict[str, Any],
        diagnostic_data: torch.Tensor,
        normalization: LatentAffineNormalization | None = None,
        prior: LatentGaussianPrior | None = None,
    ) -> None:
        self.args = args
        self.run_dir = run_dir
        self.cache_metadata = cache_metadata
        self.diagnostic_data = diagnostic_data
        self.normalization = normalization or LatentAffineNormalization.identity()
        self.prior = prior or LatentGaussianPrior.standard()
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1_000_003)
        prior_sampler = self.prior.sampler(device="cpu", dtype=torch.float32)
        fixed_z = torch.randn(args.sample_count, CELEBA_LATENT_DIM, generator=generator)
        fixed_z_ref = torch.randn(args.sample_count, CELEBA_LATENT_DIM, generator=generator)
        self.fixed_z = prior_sampler.transform_standard(fixed_z, rowwise=True).to(args.device)
        self.fixed_z_ref = prior_sampler.transform_standard(fixed_z_ref, rowwise=True).to(args.device)
        self._codec: CelebALatentCodec | None = None

    def _vae_dtype(self) -> torch.dtype:
        if self.args.vae_dtype == "float32":
            return torch.float32
        if self.args.vae_dtype == "float16":
            return torch.float16
        if self.args.vae_dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16 if self.args.device.type == "cuda" else torch.float32

    def _get_codec(self) -> CelebALatentCodec:
        if self._codec is None:
            scaling_factor = (
                self.args.vae_scaling_factor
                if self.args.vae_scaling_factor is not None
                else float(self.cache_metadata["scaling_factor"])
            )
            self._codec = CelebALatentCodec.from_pretrained(
                str(self.args.vae_model.resolve()),
                scaling_factor=scaling_factor,
                device=self.args.device,
                torch_dtype=self._vae_dtype(),
                local_files_only=True,
            )
        return self._codec

    def _decode(self, flat_latents: torch.Tensor) -> torch.Tensor:
        codec = self._get_codec()
        flat_latents = self.normalization.denormalize(flat_latents)
        chunks = [
            codec.decode_flat(chunk.to(self.args.device)).cpu()
            for chunk in flat_latents.split(self.args.decode_batch)
        ]
        return torch.cat(chunks)

    @torch.inference_mode()
    def generate(self, model: nn.Module, tag: str, logger: JsonlLogger) -> dict[str, Any]:
        was_training = model.training
        model.eval()
        artifact_directory = self.run_dir / "artifacts" / tag
        artifact_directory.mkdir(parents=True, exist_ok=True)
        samples_device = {
            nfe: _sample_model(
                model,
                self.args.method,
                self.fixed_z,
                nfe,
                self.args.sample_time_power,
            )
            for nfe in SAMPLE_NFES
        }
        reference_samples = {
            nfe: _sample_model(
                model,
                self.args.method,
                self.fixed_z_ref,
                nfe,
                self.args.sample_time_power,
            )
            for nfe in SAMPLE_NFES
        }
        samples = {nfe: tensor.detach().cpu() for nfe, tensor in samples_device.items()}

        sample_metrics = _latent_sample_metrics(samples)
        wgf_rows = []
        diagnostic_data = self.diagnostic_data.to(self.args.device)
        for nfe in SAMPLE_NFES:
            diagnostics = wgf_velocity_diagnostics(
                self.args.velocity,
                samples_device[nfe],
                diagnostic_data,
                reference_samples[nfe],
                self.args.bandwidth,
                rbf_geometry=self.args.rbf_geometry,
            )
            wgf_rows.append(
                {"nfe": nfe, **{key: float(value.detach().cpu()) for key, value in diagnostics.items()}}
            )

        transition_rows = (
            compute_transition_metrics(model, self.fixed_z, tuple(self.args.transition_times))
            if self.args.method == "flow"
            else []
        )
        metrics = {
            "tag": tag,
            "method": self.args.method,
            "sample_metrics": sample_metrics,
            "wgf_diagnostics": wgf_rows,
            "transition_metrics": transition_rows,
            "diagnostic_data_count": diagnostic_data.shape[0],
        }
        if self.prior.mode != "standard":
            metrics["latent_prior"] = self.prior.metadata()
        atomic_json_dump(artifact_directory / "metrics.json", metrics)
        if self.args.save_latent_samples:
            latent_payload: dict[str, Any] = {
                "prior": self.fixed_z.detach().cpu(),
                "samples": samples,
            }
            if self.prior.mode != "standard":
                latent_payload["prior_metadata"] = self.prior.metadata()
            atomic_torch_save(
                artifact_directory / "latents.pt",
                latent_payload,
            )
        if self.args.decode_samples:
            decoded = {nfe: self._decode(sample) for nfe, sample in samples.items()}
            save_decoded_panel(
                decoded,
                artifact_directory / "decoded_nfe_grid.png",
                title=f"{tag}: {self.args.method}",
            )
            reference_path = self.run_dir / "decoded_training_reference.png"
            if not reference_path.exists():
                save_decoded_panel(
                    {0: self._decode(self.diagnostic_data[: self.args.sample_count])},
                    reference_path,
                    title="deterministic VAE training latents",
                )

        transition_nontrivial = [
            row["relative_rmse"]
            for row in transition_rows
            if row["end_time"] > row["start_time"]
        ]
        logger.log(
            {
                "event": "artifacts",
                "tag": tag,
                "method": self.args.method,
                "max_nfe_latent_std": sample_metrics[-1]["latent_std"],
                "mean_transition_relative_rmse": (
                    sum(transition_nontrivial) / len(transition_nontrivial)
                    if transition_nontrivial
                    else None
                ),
                "artifact_directory": str(artifact_directory),
            }
        )
        model.train(was_training)
        return metrics


def write_run_metadata(
    args: argparse.Namespace,
    run_dir: Path,
    cache_metadata: dict[str, Any],
    cache_latents: torch.Tensor,
    train_start: int,
    train_end: int,
    model: nn.Module,
    normalization: LatentAffineNormalization,
    prior: LatentGaussianPrior,
) -> None:
    cache_sha256 = sha256_file(args.cache)
    if prior.mode != "standard" and prior.source_cache_sha256 != cache_sha256:
        raise RuntimeError("latent cache changed between train-only prior fit and metadata write")
    snapshot_name = (
        "source_snapshot"
        if args.resume is None
        else f"source_snapshot_resume_{utc_now().replace(':', '-')}"
    )
    source_snapshot = archive_source_snapshot(run_dir, snapshot_name)
    config = {key: value for key, value in vars(args).items() if not key.startswith("_")}
    config.update(
        {
            "latent_dim": CELEBA_LATENT_DIM,
            "prior": "N(0,I)" if prior.mode == "standard" else prior.mode,
            "sample_nfes": SAMPLE_NFES,
            "train_start_resolved": train_start,
            "train_end_resolved": train_end,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "latent_normalization_stats": normalization.metadata(
                train_start_index=train_start,
                train_end_index=train_end,
            ),
        }
    )
    cache_vae_hashes = cache_metadata.get("vae_files_sha256")
    if cache_vae_hashes is not None:
        config["cache_vae_files_sha256"] = cache_vae_hashes
    if prior.mode != "standard":
        sidecar = run_dir / LATENT_PRIOR_FILENAME
        if not sidecar.is_file():
            raise FileNotFoundError(f"latent-prior sidecar is missing: {sidecar}")
        config.update(
            {
                "latent_prior_sidecar": LATENT_PRIOR_FILENAME,
                "latent_prior_sidecar_sha256": sha256_file(sidecar),
                "latent_prior_stats": prior.metadata(),
            }
        )
    config_name = "config.json" if args.resume is None else f"config_resume_{utc_now().replace(':', '-')}.json"
    atomic_json_dump(run_dir / config_name, config)
    selected = cache_latents[train_start:train_end]
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
        "device_name": (
            torch.cuda.get_device_name(args.device) if args.device.type == "cuda" else platform.processor()
        ),
        "world_size": args.world_size,
        "global_batch": args.global_batch,
        "slurm": {
            key: os.environ.get(key)
            for key in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_NODELIST")
        },
        "git": git_provenance(),
        "source_snapshot": source_snapshot,
        "cache": {
            "path": str(args.cache.resolve()),
            "sha256": cache_sha256,
            "metadata": cache_metadata,
            "selected_mean": float(selected.mean()),
            "selected_std": float(selected.std()),
            "selected_rms": float(selected.square().mean().sqrt()),
        },
        "latent_normalization": normalization.metadata(
            train_start_index=train_start,
            train_end_index=train_end,
        ),
        "latent_prior": prior.metadata(),
    }
    if getattr(args, "feature_checkpoint", None) is not None:
        provenance["drifting_feature_checkpoint"] = {
            "path": str(args.feature_checkpoint.resolve()),
            "sha256": sha256_file(args.feature_checkpoint),
            "frozen_during_generator_training": True,
        }
    if getattr(args, "initial_model_checkpoint", None) is not None:
        provenance["initial_model_checkpoint"] = {
            "path": str(args.initial_model_checkpoint.resolve()),
            "sha256": sha256_file(args.initial_model_checkpoint),
            "weight_kind": args.initial_weight_kind,
            "load_transform": getattr(args, "_initial_model_load_transform", None),
            "optimizer_state_imported": False,
            "ema_state_imported": False,
            "data_stream_state_imported": False,
        }
    if getattr(args, "endpoint_teacher_checkpoint", None) is not None:
        provenance["endpoint_teacher_checkpoint"] = {
            "path": str(args.endpoint_teacher_checkpoint.resolve()),
            "sha256": sha256_file(args.endpoint_teacher_checkpoint),
            "weight_kind": args.endpoint_teacher_weight_kind,
            "load_transform": getattr(args, "_endpoint_teacher_load_transform", None),
            "frozen_during_flow_training": True,
        }
    if prior.mode != "standard":
        provenance["latent_prior"]["sidecar"] = LATENT_PRIOR_FILENAME
        provenance["latent_prior"]["sidecar_sha256"] = sha256_file(
            run_dir / LATENT_PRIOR_FILENAME
        )
    provenance_name = (
        "provenance.json"
        if args.resume is None
        else f"provenance_resume_{utc_now().replace(':', '-')}.json"
    )
    atomic_json_dump(run_dir / provenance_name, provenance)
    try:
        environment = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        environment = "unavailable\n"
    (run_dir / "environment.txt").write_text(environment, encoding="utf-8")


def train(args: argparse.Namespace) -> Path:
    cache_latents, cache_metadata = load_latent_cache(args.cache)
    cache_vae_hashes = cache_metadata.get("vae_files_sha256")
    if args.decode_samples and cache_vae_hashes is not None:
        def validate_decoder_vae() -> None:
            if directory_file_hashes(args.vae_model) != cache_vae_hashes:
                raise ValueError(
                    "decoder VAE files differ from the encoder attested by the cache"
                )

        run_on_rank_zero(args, validate_decoder_vae)
    smoke_count = max(args.batch * args.world_size * 2, 8)
    if args.latent_prior == "train-full-gaussian":
        # A barely square centered covariance is numerically fragile; the
        # deterministic smoke path keeps two observations per dimension.
        smoke_count = max(smoke_count, 2 * CELEBA_LATENT_DIM)
    raw_training_latents, train_start, train_end = select_training_latents(
        cache_latents,
        start_index=args.train_start_index,
        end_index=args.train_end_index,
        smoke=args.smoke,
        smoke_count=smoke_count,
    )
    prior = fit_training_prior(args, raw_training_latents, train_start, train_end)
    normalization = (
        LatentAffineNormalization.fit_channel(
            raw_training_latents,
            train_start_index=train_start,
            train_end_index=train_end,
        )
        if args.latent_normalization == "channel"
        else LatentAffineNormalization.identity()
    )
    training_latents = prepare_training_latents(raw_training_latents, normalization)
    prior_sampler = prior.sampler(device=args.device, dtype=training_latents.dtype)
    minimum_count = args.batch * args.world_size
    if training_latents.shape[0] < minimum_count:
        raise ValueError(
            f"selected {training_latents.shape[0]} latents but at least {minimum_count} are required"
        )

    stream = LatentBatchStream(training_latents, args)
    model = make_model(args).to(args.device)
    if args.initial_model_checkpoint is not None:
        args._initial_model_load_transform = load_initial_model_weights(
            model,
            args.initial_model_checkpoint,
            weight_kind=args.initial_weight_kind,
        )
    endpoint_teacher = make_endpoint_teacher(args)
    feature_encoder = make_drifting_feature_encoder(args, cache_metadata)
    raw_model = model
    if args.distributed:
        raw_model = DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    # Rank-dependent stochastic samples begin only after identical initialization.
    torch.manual_seed(args.seed + args.rank)
    if args.device.type == "cuda":
        torch.cuda.manual_seed(args.seed + args.rank)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
    )
    ema = make_ema(raw_model, args)
    ema_started = bool(ema is not None and args.ema_warmup_steps == 0)
    signature = training_signature(
        args,
        cache_metadata,
        train_start,
        train_end,
        normalization,
        prior,
        cache_sha256=sha256_file(args.cache),
        train_values_sha256=sha256_tensor(unflatten_latents(raw_training_latents)),
    )
    start_step = 0
    if args.resume is not None:
        start_step, ema_started = load_training_checkpoint(
            args.resume,
            model=raw_model,
            optimizer=optimizer,
            ema=ema,
            stream=stream,
            expected_signature=signature,
            expected_prior=prior,
            args=args,
        )
    if start_step > args.steps:
        raise ValueError(f"checkpoint is at step {start_step}, beyond requested --steps {args.steps}")

    run_dir = resolve_run_directory(args)
    run_on_rank_zero(args, lambda: ensure_latent_prior_sidecar(run_dir, prior))
    logger = JsonlLogger(run_dir / "metrics.jsonl")
    run_on_rank_zero(
        args,
        lambda: write_run_metadata(
            args,
            run_dir,
            cache_metadata,
            cache_latents,
            train_start,
            train_end,
            unwrap_model(raw_model),
            normalization,
            prior,
        ),
    )
    if is_main_process(args):
        logger.log(
            {
                "event": "start" if start_step == 0 else "resume",
                "step": start_step,
                "run_dir": str(run_dir),
                "train_count": training_latents.shape[0],
                "global_batch": args.global_batch,
                "model_parameters": sum(parameter.numel() for parameter in raw_model.parameters()),
            }
        )

    diagnostic_indices = torch.linspace(
        0,
        training_latents.shape[0] - 1,
        min(args.diagnostic_count, training_latents.shape[0]),
    ).long()
    diagnostic_data = training_latents[diagnostic_indices].clone()
    artifact_generator = (
        ArtifactGenerator(args, run_dir, cache_metadata, diagnostic_data, normalization, prior)
        if is_main_process(args)
        else None
    )

    started_at = time.monotonic()
    progress = trange(
        start_step,
        args.steps,
        initial=start_step,
        total=args.steps,
        desc=f"celeba-latent:{args.method}",
        disable=not is_main_process(args),
    )
    for step_index in progress:
        warmup_steps = getattr(args, "lr_warmup_steps", 0)
        learning_rate = args.lr * (
            min((step_index + 1) / warmup_steps, 1.0)
            if warmup_steps > 0
            else 1.0
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        data = stream.next().to(args.device, non_blocking=args.device.type == "cuda")
        loss, local_stats = training_loss(
            raw_model,
            data,
            args,
            step_index,
            prior_sampler,
            feature_encoder,
            endpoint_teacher,
        )
        finite = torch.isfinite(loss.detach()).to(dtype=torch.int32)
        if args.distributed:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError(f"non-finite loss at step {step_index + 1}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(),
            args.grad_clip if args.grad_clip > 0 else float("inf"),
            error_if_nonfinite=True,
        )
        optimizer.step()
        completed_step = step_index + 1
        ema_started = update_ema(
            ema,
            unwrap_model(raw_model),
            completed_step=completed_step,
            warmup_steps=args.ema_warmup_steps,
            started=ema_started,
        )

        should_log = (
            completed_step % args.log_every == 0
            or completed_step == 1
            or completed_step == args.steps
        )
        if should_log:
            reduced = reduce_stats({**local_stats, "grad_norm": grad_norm}, args)
            record = {
                "event": "train",
                "step": completed_step,
                "epoch": stream.epoch,
                "batch_offset": stream.batch_offset,
                "elapsed_seconds": time.monotonic() - started_at,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "distill_span": distillation_span(args, step_index),
                **{key: float(value.detach().cpu()) for key, value in reduced.items()},
                **weighted_loss_stats(reduced, args),
            }
            if is_main_process(args):
                logger.log(record)
                progress.set_postfix(
                    loss=f"{record.get('loss', float(loss.detach())):.3e}",
                    velocity=f"{record.get('velocity_norm', 0.0):.2e}",
                )

        if args.checkpoint_every and completed_step % args.checkpoint_every == 0:
            save_training_checkpoint(
                run_dir / "checkpoints" / f"checkpoint_step_{completed_step:08d}.pt",
                completed_step=completed_step,
                model=raw_model,
                optimizer=optimizer,
                ema=ema,
                ema_started=ema_started,
                stream=stream,
                signature=signature,
                normalization=normalization,
                prior=prior,
                run_dir=run_dir,
                args=args,
            )
        if args.sample_every and completed_step % args.sample_every == 0:
            evaluation_model = ema.module if ema is not None and ema_started else unwrap_model(raw_model)
            assert artifact_generator is not None or not is_main_process(args)
            run_on_rank_zero(
                args,
                lambda: artifact_generator.generate(evaluation_model, f"step_{completed_step:08d}", logger),
            )

    save_training_checkpoint(
        run_dir / "checkpoint_final.pt",
        completed_step=args.steps,
        model=raw_model,
        optimizer=optimizer,
        ema=ema,
        ema_started=ema_started,
        stream=stream,
        signature=signature,
        normalization=normalization,
        prior=prior,
        run_dir=run_dir,
        args=args,
    )

    def save_final_outputs() -> None:
        nonlocal ema_started
        unwrapped = unwrap_model(raw_model)
        atomic_torch_save(run_dir / "model_raw_final.pt", unwrapped.state_dict())
        assert artifact_generator is not None
        artifact_generator.generate(unwrapped, "final_raw", logger)
        if ema is not None:
            if not ema_started:
                ema.module.load_state_dict(unwrapped.state_dict())
                ema_started = True
            atomic_torch_save(run_dir / "model_ema_final.pt", ema.module.state_dict())
            artifact_generator.generate(ema.module, "final_ema", logger)
        logger.log(
            {
                "event": "complete",
                "step": args.steps,
                "elapsed_seconds": time.monotonic() - started_at,
                "run_dir": str(run_dir),
            }
        )

    run_on_rank_zero(args, save_final_outputs)
    return run_dir


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    setup_distributed(args)
    try:
        seed_process(args)
        return train(args)
    finally:
        cleanup_distributed(args)


if __name__ == "__main__":
    main()
