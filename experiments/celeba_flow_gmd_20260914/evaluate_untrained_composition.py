#!/usr/bin/env python3
"""Official-validation diagnostic for residual composition of a GMD parent.

This is deliberately a zero-training control.  It converts a validated
one-step GMD checkpoint to the identity-grounded two-time parameterisation in
``src.celeba_gmd_flow_map`` and composes it for NFE 1/2/4/8/16.  The real
reference, sample-index RNG, uint8 quantisation, Inception weights, FID and KID
are identical to ``experiments/celeba_baseline_20260911/evaluate.py``.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

cache_base = os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE")
if cache_base:
    cache = Path(cache_base) / os.environ.get("LOCAL_RANK", "0")
    for directory in (cache, cache / "userdb", cache / "kernels"):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        MIOPEN_CACHE_DIR=str(cache),
        MIOPEN_USER_DB_PATH=str(cache / "userdb"),
        MIOPEN_CUSTOM_CACHE_DIR=str(cache / "kernels"),
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
from src.celeba_gmd_flow_map import load_one_step_state, sample_two_time_flow


PARENT_60K = ROOT / (
    "runs/celeba_baseline_20260911/gmd_sharded_mae82_convnext13_seed42/"
    "checkpoints/checkpoint_step_00060000.pt"
)
PARENT_60K_SHA256 = "177398ff98b38c8ce3e45287a1804bac90243d75d8d7154fc85775ed33f84e8d"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PARENT_60K)
    parser.add_argument("--checkpoint-sha256", default=PARENT_60K_SHA256)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=("ema", "model"), default="ema")
    parser.add_argument("--nfe", type=int, choices=(1, 2, 4, 8, 16), required=True)
    parser.add_argument("--time-power", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=19_867)
    parser.add_argument("--seed", type=int, default=42_001)
    parser.add_argument("--generation-batch", type=int, default=64)
    parser.add_argument("--inception-batch", type=int, default=128)
    parser.add_argument("--precision", choices=("bf16",), default="bf16")
    parser.add_argument("--kid-repetitions", type=int, default=100)
    args = parser.parse_args(argv)
    if args.checkpoint.resolve() == PARENT_60K.resolve() and (
        args.checkpoint_sha256 != PARENT_60K_SHA256
    ):
        raise ValueError("the pinned 60k parent requires its canonical SHA-256")
    if args.num_samples < 2 or args.kid_repetitions < 2:
        raise ValueError("FID/KID require at least two samples and two KID repetitions")
    if min(args.generation_batch, args.inception_batch) < 1:
        raise ValueError("batch sizes must be positive")
    if not np.isfinite(args.time_power) or args.time_power <= 0:
        raise ValueError("time-power must be finite and positive")
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


def load_flow(args: argparse.Namespace, device: torch.device):
    if not args.checkpoint.is_file() or args.checkpoint.is_symlink():
        raise FileNotFoundError("checkpoint must be a regular non-symlink file")
    observed = sha256_file(args.checkpoint)
    if observed != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA-256 differs")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    flow = load_one_step_state(payload, weights=args.weights)
    flow.to(device).eval().requires_grad_(False)
    return flow, int(payload["completed_step"]), payload["training_signature"], observed


@torch.inference_mode()
def generate(
    flow,
    args: argparse.Namespace,
    start: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    noise, labels = baseline.row_noise(
        start,
        count,
        args.seed,
        flow.noise_classes,
        flow.noise_coords,
    )
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=True):
        images = sample_two_time_flow(
            flow,
            noise.to(device),
            labels.to(device),
            nfe=args.nfe,
            time_power=args.time_power,
        )
    if not bool(torch.isfinite(images).all()):
        raise FloatingPointError("non-finite generated images")
    images = images.float().cpu()
    images = (images.clamp(-1, 1) + 1.0) * 0.5
    return images.clamp(0, 1).mul(255).round().to(torch.uint8)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    world, rank, local_rank = distributed_coordinates()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    if not torch.cuda.is_available():
        raise RuntimeError("the official composition evaluation requires ROCm GPUs")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))

    if rank == 0:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        for path, expected in (
            (baseline.REFERENCE, baseline.REFERENCE_SHA256),
            (baseline.INCEPTION, baseline.INCEPTION_SHA256),
        ):
            if sha256_file(path) != expected:
                raise ValueError(f"reference SHA-256 differs: {path}")
        atomic_write_json(
            args.output / "started.json",
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": args.checkpoint_sha256,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "world_size": world,
                "nfe": args.nfe,
                "time_power": args.time_power,
                "unix_time": time.time(),
            },
        )
    if world > 1:
        dist.barrier()

    flow, step, training_config, checkpoint_sha = load_flow(args, device)
    extractor = build_inception_extractor(
        device, weights_path=baseline.INCEPTION
    ).requires_grad_(False)
    indices: list[int] = []
    features: list[np.ndarray] = []
    image_hashes: list[str] = []
    grid_pixels: list[tuple[int, np.ndarray]] = []
    begin = time.monotonic()
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
                    flow,
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
                    "elapsed_seconds": time.monotonic() - begin,
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
            else np.empty((0, 2048), np.float32)
        ),
        pixel_sha256=np.asarray(image_hashes, dtype="U64"),
    )
    atomic_write_json(
        args.output / f"timing_rank{rank:02d}.json",
        {
            "generation_seconds": generation_seconds,
            "inception_seconds": inception_seconds,
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
        generated = np.empty((args.num_samples, 2048), np.float32)
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

        real = np.load(baseline.REFERENCE, mmap_mode="r")
        if real.shape != (19_867, 2048):
            raise ValueError("official validation reference shape differs")
        real_mean, real_cov = streaming_mean_covariance(real)
        generated_mean, generated_cov = streaming_mean_covariance(generated)
        fid = calculate_fid(real_mean, real_cov, generated_mean, generated_cov)
        kid = repeated_kid(
            real,
            generated,
            subset_size=1_000,
            repetitions=args.kid_repetitions,
            seed=0,
        )
        if sha256_file(args.checkpoint) != checkpoint_sha:
            raise RuntimeError("checkpoint changed during evaluation")
        result = {
            "kind": "gmd_two_time_untrained_composition_control",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "parent_step": step,
            "weights": args.weights,
            "nfe": args.nfe,
            "time_power": args.time_power,
            "fid": fid,
            "kid": kid,
            "num_generated": args.num_samples,
            "num_real": len(real),
            "seed": args.seed,
            "reference": str(baseline.REFERENCE),
            "reference_sha256": baseline.REFERENCE_SHA256,
            "inception_sha256": baseline.INCEPTION_SHA256,
            "inception_batch": args.inception_batch,
            "generation_batch": args.generation_batch,
            "precision": args.precision,
            "quantization": (
                "FP32 clamp(-1,1), map to [0,1], uint8 round; FP32/255 Inception"
            ),
            "unique_generated_pixel_hashes": len(set(hashes.tolist())),
            "feature_trace": float(np.trace(generated_cov)),
            "world_size": world,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "elapsed_seconds": time.monotonic() - begin,
            "source_sha256": sha256_file(Path(__file__)),
            "flow_source_sha256": sha256_file(
                ROOT / "src/celeba_gmd_flow_map.py"
            ),
            "backbone_source_sha256": sha256_file(
                ROOT / "src/celeba_pixel_gmd_generator.py"
            ),
            "baseline_evaluator_source_sha256": sha256_file(
                ROOT / "experiments/celeba_baseline_20260911/evaluate.py"
            ),
            "training_config": training_config,
            "flow_training_updates": 0,
            "composition": (
                "fixed labels; no reinjected noise; uniform partition; "
                "endpoint interpolation (t-s)/(1-s)"
            ),
            "protocol": (
                "official validation development; sample-count matched"
                if args.num_samples == len(real)
                else "diagnostic only: generated and reference counts differ"
            ),
        }
        atomic_write_json(args.output / "metrics.json", result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in ("kind", "parent_step", "nfe", "fid", "num_generated")
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
