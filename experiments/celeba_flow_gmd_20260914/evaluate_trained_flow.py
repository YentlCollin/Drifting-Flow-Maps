#!/usr/bin/env python3
"""Official-validation FID/KID for a trained CelebA two-time GMD flow.

The primary input is the strict, self-describing inference artifact emitted for
``ComposedRollout``.  Sampling deliberately matches
``evaluate_untrained_composition.py``: row-indexed noise and labels, fixed
labels for every transition, no noise reinjection, a final FP32 clip and uint8
round-trip, and the same pinned Inception/reference FID and KID protocol.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# torchrun defines LOCAL_RANK before importing this module.  Keep every cache
# rank-local; sharing MIOpen or compiler caches between eight processes can
# corrupt first-use compilation state on ROCm.
cache_base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
if cache_base:
    cache = Path(cache_base) / os.environ.get("LOCAL_RANK", "0")
    cache_directories = {
        "miopen": cache,
        "miopen_userdb": cache / "userdb",
        "miopen_kernels": cache / "kernels",
        "matplotlib": cache / "matplotlib",
        "torch": cache / "torch",
        "torchinductor": cache / "torchinductor",
        "triton": cache / "triton",
    }
    for directory in cache_directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        MIOPEN_CACHE_DIR=str(cache_directories["miopen"]),
        MIOPEN_USER_DB_PATH=str(cache_directories["miopen_userdb"]),
        MIOPEN_CUSTOM_CACHE_DIR=str(cache_directories["miopen_kernels"]),
        MPLCONFIGDIR=str(cache_directories["matplotlib"]),
        PYTORCH_KERNEL_CACHE_PATH=str(cache_directories["torch"]),
        TORCHINDUCTOR_CACHE_DIR=str(cache_directories["torchinductor"]),
        TRITON_CACHE_DIR=str(cache_directories["triton"]),
    )

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist

from experiments.celeba_baseline_20260911 import evaluate as baseline
from experiments.evaluate_image_directories import (
    atomic_write_json,
    build_inception_extractor,
    calculate_fid,
    repeated_kid,
    sha256_file,
    streaming_mean_covariance,
)
from src.celeba_gmd_composed_rollout import (
    ARTIFACT_FORMAT,
    ARTIFACT_VERSION,
    load_composed_rollout_artifact,
)
from src.celeba_gmd_flow_map import sample_two_time_flow
from src.celeba_pixel_gmd_features import canonical_json_sha256


SUPPORTED_NFES = (1, 2, 4, 8, 16)
OFFICIAL_NUM_SAMPLES = 19_867
INCEPTION_FEATURE_DIMENSION = 2_048
KID_SUBSET_SIZE = 1_000
_SHA256 = re.compile(r"[0-9a-f]{64}")

SOURCE_RELATIVE_PATHS = (
    'experiments/celeba_flow_gmd_20260914/evaluate_trained_flow.py',
    'experiments/celeba_flow_gmd_20260914/evaluate_untrained_composition.py',
    'experiments/celeba_baseline_20260911/evaluate.py',
    'experiments/evaluate_image_directories.py',
    'src/celeba_gmd_composed_rollout.py',
    'src/celeba_gmd_flow_map.py',
    'src/celeba_gmd_flow_objectives.py',
    'src/celeba_pixel_gmd_features.py',
    'src/celeba_pixel_gmd_generator.py',
)


def argparse_sha256(value: str) -> str:
    """Accept one explicit lowercase whole-file SHA-256 digest."""

    if _SHA256.fullmatch(value) is None:
        raise argparse.ArgumentTypeError(
            "SHA-256 must contain exactly 64 lowercase hexadecimal characters"
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        "--checkpoint",
        dest="checkpoint",
        type=Path,
        required=True,
        help="strict composed-rollout inference artifact",
    )
    parser.add_argument(
        "--artifact-sha256",
        "--checkpoint-sha256",
        "--input-sha256",
        dest="checkpoint_sha256",
        type=argparse_sha256,
        required=True,
        help="required SHA-256 of the complete artifact file",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nfe", type=int, choices=SUPPORTED_NFES, required=True)
    parser.add_argument(
        "--grid",
        choices=("power", "uniform"),
        default="power",
        help="power uses u**time_power; uniform requires time_power=1",
    )
    parser.add_argument("--time-power", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=OFFICIAL_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=42_001)
    parser.add_argument("--generation-batch", type=int, default=64)
    parser.add_argument("--inception-batch", type=int, default=128)
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--kid-repetitions", type=int, default=100)
    args = parser.parse_args(argv)
    if args.num_samples < 2 or args.kid_repetitions < 2:
        raise ValueError("FID/KID require at least two samples and two KID repetitions")
    if min(args.generation_batch, args.inception_batch) < 1:
        raise ValueError("batch sizes must be positive")
    if not math.isfinite(args.time_power) or args.time_power <= 0:
        raise ValueError("time-power must be finite and positive")
    if args.grid == "uniform" and args.time_power != 1.0:
        raise ValueError("the uniform grid requires --time-power 1")
    return args


def distributed_coordinates() -> tuple[int, int, int]:
    return tuple(
        int(os.environ.get(key, default))
        for key, default in (
            ("WORLD_SIZE", "1"),
            ("RANK", "0"),
            ("LOCAL_RANK", "0"),
        )
    )


def accelerator_device(local_rank: int) -> torch.device:
    """Return the rank-local accelerator (a seam for CPU protocol tests)."""

    return torch.device("cuda", local_rank)


def _regular_nonsymlink(path: Path, *, label: str) -> Path:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"{label} must be a regular non-symlink file")
    return path.resolve()


def verify_artifact_file(path: Path, expected_sha256: str) -> tuple[Path, str]:
    """Resolve and authenticate the strict inference artifact before use."""

    resolved = _regular_nonsymlink(path, label="artifact")
    observed = sha256_file(resolved)
    if observed != expected_sha256:
        raise ValueError("artifact SHA-256 differs")
    return resolved, observed


def source_inventory() -> dict[str, dict[str, Any]]:
    """Hash every source file that can affect this evaluation."""

    inventory: dict[str, dict[str, Any]] = {}
    for relative in SOURCE_RELATIVE_PATHS:
        path = _regular_nonsymlink(ROOT / relative, label=f"source {relative}")
        inventory[relative] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return inventory


def _validated_optional_nonnegative_int(
    metadata: Mapping[str, Any], names: tuple[str, ...]
) -> int | None:
    observed: list[tuple[str, int]] = []
    for name in names:
        if name not in metadata:
            continue
        value = metadata[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"artifact metadata {name!r} must be a non-negative integer")
        observed.append((name, value))
    if not observed:
        return None
    if len({value for _, value in observed}) != 1:
        raise ValueError("artifact metadata training-step fields disagree")
    return observed[0][1]


def _validated_optional_weight_kind(metadata: Mapping[str, Any]) -> str | None:
    if "weight_kind" not in metadata:
        return None
    value = metadata["weight_kind"]
    if value not in {"raw", "ema"}:
        raise ValueError("artifact metadata 'weight_kind' must be 'raw' or 'ema'")
    return value


def _validated_training_lineage(
    metadata: Mapping[str, Any], architecture: Mapping[str, Any]
) -> dict[str, Any]:
    """Cross-check the optional trainer signature embedded by the exporter."""

    signature = metadata.get("training_signature")
    expected_digest = metadata.get("training_signature_sha256")
    if signature is None and expected_digest is None:
        return {
            "training_signature": None,
            "training_signature_sha256": None,
            "training_source_files_sha256": None,
            "training_source_hashes_key": None,
            "parent": None,
        }
    if not isinstance(signature, Mapping) or not isinstance(expected_digest, str):
        raise ValueError("artifact training signature and SHA-256 must appear together")
    if _SHA256.fullmatch(expected_digest) is None or (
        canonical_json_sha256(dict(signature)) != expected_digest
    ):
        raise ValueError("artifact training-signature SHA-256 differs")
    signed_architecture = signature.get("flow")
    if not isinstance(signed_architecture, Mapping) or (
        canonical_json_sha256(dict(signed_architecture))
        != canonical_json_sha256(dict(architecture))
    ):
        raise ValueError("artifact architecture differs from its training signature")
    canonical_source_hashes = signature.get("source_files_sha256")
    structural_source_hashes = signature.get("source_and_artifact_sha256")
    if canonical_source_hashes is not None and structural_source_hashes is not None:
        if canonical_source_hashes != structural_source_hashes:
            raise ValueError(
                "artifact training signature has conflicting source-hash mappings"
            )
        source_hashes = canonical_source_hashes
        source_hashes_key = "source_files_sha256"
    elif canonical_source_hashes is not None:
        source_hashes = canonical_source_hashes
        source_hashes_key = "source_files_sha256"
    else:
        source_hashes = structural_source_hashes
        source_hashes_key = "source_and_artifact_sha256"
    if not isinstance(source_hashes, Mapping) or not source_hashes:
        raise ValueError("artifact training signature lacks source-file hashes")
    if any(
        not isinstance(name, str)
        or not isinstance(digest, str)
        or _SHA256.fullmatch(digest) is None
        for name, digest in source_hashes.items()
    ):
        raise ValueError("artifact training source-file hashes are malformed")
    parent = signature.get("initialization")
    if not isinstance(parent, Mapping):
        raise ValueError("artifact training signature lacks parent lineage")
    return {
        "training_signature": copy.deepcopy(dict(signature)),
        "training_signature_sha256": expected_digest,
        "training_source_files_sha256": copy.deepcopy(dict(source_hashes)),
        "training_source_hashes_key": source_hashes_key,
        "parent": copy.deepcopy(dict(parent)),
    }


def load_flow(
    args: argparse.Namespace,
    device: torch.device,
):
    """Load the strict artifact without falling back to a baseline checkpoint."""

    artifact_path, observed = verify_artifact_file(
        args.checkpoint, args.checkpoint_sha256
    )
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    rollout, metadata = load_composed_rollout_artifact(payload, device)
    if sha256_file(artifact_path) != observed:
        raise RuntimeError("artifact changed while it was being loaded")

    # The strict loader has authenticated these mappings.  Copy them into the
    # result provenance so the evaluated state and architecture are explicit.
    if not isinstance(payload, Mapping):  # defensive; strict loader also checks
        raise ValueError("artifact payload is not a mapping")
    architecture = dict(payload["architecture"])
    artifact_provenance = {
        "format": payload["format"],
        "version": payload["version"],
        "integrity": copy.deepcopy(dict(payload["integrity"])),
        "architecture": copy.deepcopy(architecture),
        "metadata": copy.deepcopy(metadata),
        "completed_step": _validated_optional_nonnegative_int(
            metadata,
            ("completed_step", "flow_training_updates", "updates"),
        ),
        "weight_kind": _validated_optional_weight_kind(metadata),
        **_validated_training_lineage(metadata, architecture),
    }
    return rollout, metadata, artifact_provenance, observed


@torch.inference_mode()
def generate(
    rollout,
    args: argparse.Namespace,
    start: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate and quantize rows using the official baseline RNG protocol."""

    flow = rollout.flow
    noise, labels = baseline.row_noise(
        start,
        count,
        args.seed,
        flow.noise_classes,
        flow.noise_coords,
    )
    effective_power = args.time_power if args.grid == "power" else 1.0
    with torch.autocast(
        device.type,
        dtype=torch.bfloat16,
        enabled=args.precision == "bf16",
    ):
        # sample_two_time_flow reuses this exact labels tensor for every map
        # transition and never injects fresh noise or clips intermediate states.
        images = sample_two_time_flow(
            flow,
            noise.to(device),
            labels.to(device),
            nfe=args.nfe,
            time_power=effective_power,
        )
    if not bool(torch.isfinite(images).all()):
        raise FloatingPointError("non-finite generated images")
    images = images.float().cpu()
    images = (images.clamp(-1, 1) + 1.0) * 0.5
    return images.clamp(0, 1).mul(255).round().to(torch.uint8)


def _create_fresh_output(path: Path) -> None:
    """Create a new output directory without reusing any prior path."""

    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    path.mkdir(parents=True, exist_ok=False)


def _load_rank_timing(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"rank timing is not an object: {path}")
    return value


def _output_hashes(output: Path, world: int) -> dict[str, str]:
    names = ["started.json", "generated_features.npy", "samples_first64.png"]
    for rank in range(world):
        names.extend((f"features_rank{rank:02d}.npz", f"timing_rank{rank:02d}.json"))
        grid = output / f"grid_rank{rank:02d}.npz"
        if grid.is_file() and not grid.is_symlink():
            names.append(grid.name)
    result: dict[str, str] = {}
    for name in names:
        path = _regular_nonsymlink(output / name, label=f"output {name}")
        result[name] = sha256_file(path)
    return result


def main(argv: list[str] | None = None) -> Path:
    program_begin = time.monotonic()
    started_unix_time = time.time()
    args = parse_args(argv)
    world, rank, local_rank = distributed_coordinates()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("the official trained-flow evaluation requires ROCm GPUs")
    device = accelerator_device(local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))

    artifact_path, artifact_sha256 = verify_artifact_file(
        args.checkpoint, args.checkpoint_sha256
    )
    sources = source_inventory()
    if rank == 0:
        _create_fresh_output(args.output)
        for path, expected in (
            (baseline.REFERENCE, baseline.REFERENCE_SHA256),
            (baseline.INCEPTION, baseline.INCEPTION_SHA256),
        ):
            if sha256_file(path) != expected:
                raise ValueError(f"reference SHA-256 differs: {path}")
        atomic_write_json(
            args.output / "started.json",
            {
                "kind": "gmd_two_time_trained_flow_evaluation",
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_sha256,
                "input": str(artifact_path),
                "input_sha256": artifact_sha256,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "world_size": world,
                "nfe": args.nfe,
                "grid": args.grid,
                "time_power": args.time_power,
                "num_samples": args.num_samples,
                "seed": args.seed,
                "source_files": sources,
                "unix_time": started_unix_time,
            },
        )
    if world > 1:
        dist.barrier()

    load_begin = time.monotonic()
    rollout, metadata, artifact_provenance, loaded_sha256 = load_flow(args, device)
    artifact_load_seconds = time.monotonic() - load_begin
    extractor_begin = time.monotonic()
    extractor = build_inception_extractor(
        device, weights_path=baseline.INCEPTION
    ).requires_grad_(False)
    inception_load_seconds = time.monotonic() - extractor_begin

    indices: list[int] = []
    features: list[np.ndarray] = []
    image_hashes: list[str] = []
    grid_pixels: list[tuple[int, np.ndarray]] = []
    worker_begin = time.monotonic()
    generation_seconds = 0.0
    inception_seconds = 0.0
    blocks = list(range(0, args.num_samples, args.inception_batch))
    for block in blocks[rank::world]:
        stop = min(args.num_samples, block + args.inception_batch)
        torch.cuda.synchronize()
        start_time = time.monotonic()
        pixels = torch.cat(
            [
                generate(
                    rollout,
                    args,
                    start,
                    min(args.generation_batch, stop - start),
                    device,
                )
                for start in range(block, stop, args.generation_batch)
            ]
        )
        torch.cuda.synchronize()
        generation_seconds += time.monotonic() - start_time
        start_time = time.monotonic()
        with torch.inference_mode():
            activation = extractor(pixels.to(device).float().div_(255))[0].flatten(1)
            feature = activation.float().cpu().numpy()
        if not np.isfinite(feature).all():
            raise FloatingPointError("non-finite Inception activations")
        inception_seconds += time.monotonic() - start_time
        indices.extend(range(block, stop))
        features.append(feature)
        for index, pixel in zip(range(block, stop), pixels, strict=True):
            image_hashes.append(hashlib.sha256(pixel.numpy().tobytes()).hexdigest())
            if index < 64:
                grid_pixels.append((index, pixel.numpy()))
        print(
            json.dumps(
                {
                    "rank": rank,
                    "generated": len(indices),
                    "last_index": stop - 1,
                    "elapsed_seconds": time.monotonic() - worker_begin,
                }
            ),
            flush=True,
        )

    np.savez(
        args.output / f"features_rank{rank:02d}.npz",
        indices=np.asarray(indices, dtype=np.int64),
        features=(
            np.concatenate(features)
            if features
            else np.empty((0, INCEPTION_FEATURE_DIMENSION), np.float32)
        ),
        pixel_sha256=np.asarray(image_hashes, dtype="U64"),
    )
    worker_elapsed_seconds = time.monotonic() - worker_begin
    atomic_write_json(
        args.output / f"timing_rank{rank:02d}.json",
        {
            "rank": rank,
            "artifact_load_seconds": artifact_load_seconds,
            "inception_load_seconds": inception_load_seconds,
            "generation_seconds": generation_seconds,
            "inception_seconds": inception_seconds,
            "worker_elapsed_seconds": worker_elapsed_seconds,
            "num_samples": len(indices),
        },
    )
    if grid_pixels:
        np.savez(
            args.output / f"grid_rank{rank:02d}.npz",
            indices=[value[0] for value in grid_pixels],
            pixels=np.stack([value[1] for value in grid_pixels]),
        )
    if world > 1:
        dist.barrier()

    if rank == 0:
        finalization_begin = time.monotonic()
        generated = np.empty(
            (args.num_samples, INCEPTION_FEATURE_DIMENSION), np.float32
        )
        covered = np.zeros(args.num_samples, np.int32)
        hashes = np.empty(args.num_samples, dtype="U64")
        for worker in range(world):
            with np.load(args.output / f"features_rank{worker:02d}.npz") as shard:
                generated[shard["indices"]] = shard["features"]
                hashes[shard["indices"]] = shard["pixel_sha256"]
                covered[shard["indices"]] += 1
        if not np.all(covered == 1):
            raise RuntimeError("incomplete or duplicate sample coverage")
        np.save(args.output / "generated_features.npy", generated)

        grid = np.zeros((8 * 64, 8 * 64, 3), np.uint8)
        for path in sorted(args.output.glob("grid_rank*.npz")):
            with np.load(path) as shard:
                for index, pixel in zip(
                    shard["indices"], shard["pixels"], strict=True
                ):
                    row, column = divmod(int(index), 8)
                    grid[
                        row * 64 : (row + 1) * 64,
                        column * 64 : (column + 1) * 64,
                    ] = pixel.transpose(1, 2, 0)
        Image.fromarray(grid).save(args.output / "samples_first64.png")

        metrics_begin = time.monotonic()
        real = np.load(baseline.REFERENCE, mmap_mode="r")
        if real.shape != (OFFICIAL_NUM_SAMPLES, INCEPTION_FEATURE_DIMENSION):
            raise ValueError("official validation reference shape differs")
        real_mean, real_cov = streaming_mean_covariance(real)
        generated_mean, generated_cov = streaming_mean_covariance(generated)
        fid = calculate_fid(real_mean, real_cov, generated_mean, generated_cov)
        kid = repeated_kid(
            real,
            generated,
            subset_size=KID_SUBSET_SIZE,
            repetitions=args.kid_repetitions,
            seed=0,
        )
        metrics_seconds = time.monotonic() - metrics_begin

        if sha256_file(artifact_path) != loaded_sha256:
            raise RuntimeError("artifact changed during evaluation")
        for path, expected in (
            (baseline.REFERENCE, baseline.REFERENCE_SHA256),
            (baseline.INCEPTION, baseline.INCEPTION_SHA256),
        ):
            if sha256_file(path) != expected:
                raise RuntimeError(f"metric input changed during evaluation: {path}")
        if source_inventory() != sources:
            raise RuntimeError("evaluation source changed during evaluation")

        rank_timings = [
            _load_rank_timing(args.output / f"timing_rank{worker:02d}.json")
            for worker in range(world)
        ]
        effective_power = args.time_power if args.grid == "power" else 1.0
        uniform_partition = effective_power == 1.0
        parent = artifact_provenance["parent"]
        parent_step = (
            parent.get("completed_step") if isinstance(parent, Mapping) else None
        )
        timing = {
            "rank": rank_timings,
            "generation_seconds_sum": sum(
                float(value["generation_seconds"]) for value in rank_timings
            ),
            "generation_seconds_max_rank": max(
                float(value["generation_seconds"]) for value in rank_timings
            ),
            "inception_seconds_sum": sum(
                float(value["inception_seconds"]) for value in rank_timings
            ),
            "inception_seconds_max_rank": max(
                float(value["inception_seconds"]) for value in rank_timings
            ),
            "metrics_seconds": metrics_seconds,
            "finalization_seconds": time.monotonic() - finalization_begin,
            "elapsed_seconds": time.monotonic() - program_begin,
        }
        result = {
            "kind": "gmd_two_time_trained_composition",
            "artifact": str(artifact_path),
            "artifact_sha256": loaded_sha256,
            "artifact_size_bytes": artifact_path.stat().st_size,
            "input": str(artifact_path),
            "input_sha256": loaded_sha256,
            # Retain baseline-compatible names for simple comparison scripts.
            "checkpoint": str(artifact_path),
            "checkpoint_sha256": loaded_sha256,
            "input_kind": "strict_composed_rollout_artifact",
            "artifact_format": ARTIFACT_FORMAT,
            "artifact_version": ARTIFACT_VERSION,
            "artifact_provenance": artifact_provenance,
            "artifact_metadata": copy.deepcopy(metadata),
            "completed_step": artifact_provenance["completed_step"],
            "step": artifact_provenance["completed_step"],
            "flow_training_updates": artifact_provenance["completed_step"],
            "weights": artifact_provenance["weight_kind"],
            "training_config": artifact_provenance["training_signature"],
            "training_signature_sha256": artifact_provenance[
                "training_signature_sha256"
            ],
            "training_source_files_sha256": artifact_provenance[
                "training_source_files_sha256"
            ],
            "training_source_hashes_key": artifact_provenance[
                "training_source_hashes_key"
            ],
            "parent": parent,
            "parent_step": parent_step,
            "nfe": args.nfe,
            "grid": args.grid,
            "time_power": args.time_power,
            "effective_time_power": effective_power,
            "fid": fid,
            "kid": kid,
            "num_generated": args.num_samples,
            "num_real": len(real),
            "seed": args.seed,
            "reference": str(baseline.REFERENCE),
            "reference_sha256": baseline.REFERENCE_SHA256,
            "inception": str(baseline.INCEPTION),
            "inception_sha256": baseline.INCEPTION_SHA256,
            "inception_batch": args.inception_batch,
            "generation_batch": args.generation_batch,
            "precision": args.precision,
            "kid_subset_size": KID_SUBSET_SIZE,
            "kid_repetitions": args.kid_repetitions,
            "kid_seed": 0,
            "quantization": (
                "FP32 clamp(-1,1), map to [0,1], uint8 round; FP32/255 Inception"
            ),
            "unique_generated_pixel_hashes": len(set(hashes.tolist())),
            "feature_trace": float(np.trace(generated_cov)),
            "world_size": world,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "started_unix_time": started_unix_time,
            "completed_unix_time": time.time(),
            "elapsed_seconds": timing["elapsed_seconds"],
            "timing": timing,
            "source_sha256": sources[
                "experiments/celeba_flow_gmd_20260914/evaluate_trained_flow.py"
            ]["sha256"],
            "flow_source_sha256": sources["src/celeba_gmd_flow_map.py"]["sha256"],
            "rollout_source_sha256": sources[
                "src/celeba_gmd_composed_rollout.py"
            ]["sha256"],
            "flow_objectives_source_sha256": sources[
                "src/celeba_gmd_flow_objectives.py"
            ]["sha256"],
            "backbone_source_sha256": sources[
                "src/celeba_pixel_gmd_generator.py"
            ]["sha256"],
            "baseline_evaluator_source_sha256": sources[
                "experiments/celeba_baseline_20260911/evaluate.py"
            ]["sha256"],
            "metric_source_sha256": sources[
                "experiments/evaluate_image_directories.py"
            ]["sha256"],
            "untrained_evaluator_source_sha256": sources[
                "experiments/celeba_flow_gmd_20260914/evaluate_untrained_composition.py"
            ]["sha256"],
            "source_files": sources,
            "composition": (
                "fixed labels; no reinjected noise; "
                + (
                    "uniform partition; "
                    if uniform_partition
                    else f"power partition u^{effective_power:g}; "
                )
                + "no intermediate clipping; endpoint interpolation (t-s)/(1-s)"
            ),
            "composition_details": {
                "labels_fixed_across_transitions": True,
                "noise_reinjected": False,
                "intermediate_clipping": False,
                "final_clipping": "FP32 [-1,1] before uint8 conversion",
                "partition": "uniform" if uniform_partition else "power",
                "power": effective_power,
            },
            "protocol": (
                "official validation development; sample-count matched"
                if args.num_samples == len(real)
                else "diagnostic only: generated and reference counts differ"
            ),
        }
        result["output_artifacts_sha256"] = _output_hashes(args.output, world)
        atomic_write_json(args.output / "metrics.json", result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "kind",
                        "completed_step",
                        "nfe",
                        "fid",
                        "num_generated",
                    )
                }
            ),
            flush=True,
        )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return args.output


if __name__ == "__main__":
    main()
