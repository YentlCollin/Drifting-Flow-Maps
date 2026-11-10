#!/usr/bin/env python3
"""Common official-validation FID/KID for pixel GMD and standard DDPM/DDIM.

Generation is quantized exactly as the historical PNG writer, then passed to
FP32 pytorch-fid. Global Inception batches are unchanged by the worker count.
Only activations, hashes, and a fixed first-64 grid are retained.
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
    os.environ.update(MIOPEN_CACHE_DIR=str(cache), MIOPEN_USER_DB_PATH=str(cache / "userdb"),
                      MIOPEN_CUSTOM_CACHE_DIR=str(cache / "kernels"))

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from experiments.evaluate_image_directories import (
    atomic_write_json, build_inception_extractor, calculate_fid,
    repeated_kid, sha256_file, streaming_mean_covariance,
)

REFERENCE = ROOT / ("runs/celeba_pixel_gmd/generator/"
    "direct_rgb_gmd_official_train_mae192ep_seed42_100k_v1/evaluation/"
    "official_validation_step00025000_raw_ema_nfe1_n19867_seed42001_v1/cache/inception/"
    "inception_aa5ab8a9f48850c8232d760b4852d4986a9ea9a8f97e75f38d0145cd6ff35025.npy")
REFERENCE_SHA256 = "0b6ed5431964eaf1591cef7d2328209077b97d8f98e0d3a7a4b85e5978bcb2b5"
INCEPTION = ROOT / "data/models/cache/pt_inception-2015-12-05-6726825d.pth"
INCEPTION_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"


def row_noise(start, count, seed, noise_classes=None, noise_coords=None):
    noise, labels = [], []
    for index in range(start, start + count):
        rng = torch.Generator().manual_seed((seed + index) % (2**63 - 1))
        noise.append(torch.randn((3, 64, 64), generator=rng))
        if noise_classes is not None:
            labels.append(torch.randint(noise_classes, (noise_coords,), generator=rng))
    return torch.stack(noise), torch.stack(labels) if labels else None


def load_model(args, device):
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.kind == "diffusion":
        from experiments.celeba_baseline_20260911.diffusion_train import build_diffusion
        model = build_diffusion(payload["config"])
        model.load_state_dict(payload[args.weights], strict=True)
        step = payload["step"]
        config = payload["config"]
    else:
        from src.celeba_pixel_gmd_generator import generator_from_architecture, load_pixel_gmd_generator_artifact
        if "training_signature" in payload:
            model = generator_from_architecture(payload["training_signature"]["generator"])
            model.load_state_dict(payload[args.weights], strict=True)
            step = payload["completed_step"]
            config = payload["training_signature"]
        else:
            model, metadata = load_pixel_gmd_generator_artifact(payload, device="cpu")
            expected_weight_kind = "raw" if args.weights == "model" else args.weights
            if metadata.get("weight_kind") != expected_weight_kind:
                raise ValueError(
                    f"GMD artifact contains {metadata.get('weight_kind')!r} weights, "
                    f"but --weights requests {expected_weight_kind!r}"
                )
            step = metadata.get("completed_step")
            config = metadata
    return model.to(device).eval().requires_grad_(False), step, config


@torch.inference_mode()
def generate(model, args, start, count, device):
    if args.kind == "gmd":
        noise, labels = row_noise(start, count, args.seed, model.noise_classes, model.noise_coords)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            images = model(noise.to(device), labels.to(device))
    else:
        from experiments.celeba_baseline_20260911.diffusion_sampling import sample_ddim
        noise, _ = row_noise(start, count, args.seed)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            images = sample_ddim(model, noise.to(device), args.nfe)
    if not torch.isfinite(images).all():
        raise FloatingPointError("nonfinite generated images")
    images = images.float().cpu()
    if args.kind == "gmd":
        # Match the historical RGB-GMD PNG writer before uint8 quantization.
        # The generator is trained in [-1,1]; the diffusion sampler already
        # returns [0,1]. Convert in FP32 after autocast, as that writer does.
        images = (images.clamp(-1, 1) + 1.0) * 0.5
    return images.clamp(0, 1).mul(255).round().to(torch.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["gmd", "diffusion"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--nfe", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=19867)
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--generation-batch", type=int, default=64)
    parser.add_argument("--inception-batch", type=int, default=128)
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--kid-repetitions", type=int, default=100)
    args = parser.parse_args()
    if args.kind == "gmd" and args.nfe != 1:
        raise ValueError("the baseline GMD is a one-call generator")
    if min(args.generation_batch, args.inception_batch, args.nfe) < 1:
        raise ValueError("positive counts required")
    if args.num_samples < 2 or args.kid_repetitions < 2:
        raise ValueError("FID/KID require at least two samples and two KID repetitions")
    world, rank, local_rank = (int(os.environ.get(key, default)) for key, default in
                              (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    if rank == 0:
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True)
        for path, expected in ((REFERENCE, REFERENCE_SHA256), (INCEPTION, INCEPTION_SHA256)):
            if sha256_file(path) != expected:
                raise ValueError(f"reference SHA256 differs: {path}")
        checkpoint_sha = sha256_file(args.checkpoint)
        atomic_write_json(args.output / "started.json", {"checkpoint_sha256": checkpoint_sha,
            "job_id": os.environ.get("SLURM_JOB_ID"), "world_size": world, "unix_time": time.time()})
    if world > 1:
        dist.barrier()
    model, step, config = load_model(args, device)
    extractor = build_inception_extractor(device, weights_path=INCEPTION).requires_grad_(False)
    indices, features, image_hashes = [], [], []
    grid_pixels = []
    begin = time.monotonic()
    generation_seconds, inception_seconds = 0., 0.
    blocks = list(range(0, args.num_samples, args.inception_batch))
    for block in blocks[rank::world]:
        stop = min(args.num_samples, block + args.inception_batch)
        torch.cuda.synchronize()
        start_time = time.monotonic()
        pixels = torch.cat([generate(model, args, start, min(args.generation_batch, stop-start), device)
                            for start in range(block, stop, args.generation_batch)])
        torch.cuda.synchronize()
        generation_seconds += time.monotonic() - start_time
        start_time = time.monotonic()
        with torch.inference_mode():
            activation = extractor(pixels.to(device).float().div_(255))[0].flatten(1)
            feature = activation.float().cpu().numpy()
        if not np.isfinite(feature).all():
            raise FloatingPointError("nonfinite Inception activations")
        inception_seconds += time.monotonic() - start_time
        indices.extend(range(block, stop))
        features.append(feature)
        for index, pixel in zip(range(block, stop), pixels):
            image_hashes.append(hashlib.sha256(pixel.numpy().tobytes()).hexdigest())
            if index < 64:
                grid_pixels.append((index, pixel.numpy()))
        print(json.dumps({"rank": rank, "generated": len(indices), "last_index": stop-1,
                          "elapsed_seconds": time.monotonic()-begin}), flush=True)
    np.savez(args.output / f"features_rank{rank:02d}.npz", indices=np.asarray(indices, dtype=np.int64),
             features=np.concatenate(features) if features else np.empty((0, 2048), np.float32),
             pixel_sha256=np.asarray(image_hashes, dtype="U64"))
    atomic_write_json(args.output / f"timing_rank{rank:02d}.json", {"generation_seconds": generation_seconds,
        "inception_seconds": inception_seconds, "num_samples": len(indices)})
    if grid_pixels:
        np.savez(args.output / f"grid_rank{rank:02d}.npz", indices=[x[0] for x in grid_pixels],
                 pixels=np.stack([x[1] for x in grid_pixels]))
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
                for index, pixel in zip(shard["indices"], shard["pixels"]):
                    row, col = divmod(int(index), 8)
                    grid[row*64:(row+1)*64, col*64:(col+1)*64] = pixel.transpose(1, 2, 0)
        Image.fromarray(grid).save(args.output / "samples_first64.png")
        real = np.load(REFERENCE, mmap_mode="r")
        if real.shape != (19867, 2048):
            raise ValueError("official validation reference shape differs")
        real_mean, real_cov = streaming_mean_covariance(real)
        gen_mean, gen_cov = streaming_mean_covariance(generated)
        fid = calculate_fid(real_mean, real_cov, gen_mean, gen_cov)
        kid = repeated_kid(real, generated, subset_size=1000, repetitions=args.kid_repetitions, seed=0)
        if sha256_file(args.checkpoint) != checkpoint_sha:
            raise RuntimeError("checkpoint changed during evaluation")
        result = {"kind": args.kind, "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha,
                  "step": step, "weights": args.weights, "nfe": args.nfe, "fid": fid, "kid": kid,
                  "num_generated": args.num_samples, "num_real": len(real), "seed": args.seed,
                  "reference": str(REFERENCE), "reference_sha256": REFERENCE_SHA256,
                  "inception_sha256": INCEPTION_SHA256, "inception_batch": args.inception_batch,
                  "generation_batch": args.generation_batch, "precision": args.precision,
                  "quantization": "GMD: (FP32 clamp(-1,1)+1)/2; diffusion: [0,1]; then clamp(0,1)*255 round uint8; FP32/255 into Inception",
                  "unique_generated_pixel_hashes": len(set(hashes.tolist())),
                  "feature_trace": float(np.trace(gen_cov)), "world_size": world,
                  "job_id": os.environ.get("SLURM_JOB_ID"), "elapsed_seconds": time.monotonic()-begin,
                  "source_sha256": sha256_file(Path(__file__)), "training_config": config,
                  "protocol": "official validation development; sample-count matched" if args.num_samples == len(real)
                              else "diagnostic only: generated and reference counts differ"}
        atomic_write_json(args.output / "metrics.json", result)
        print(json.dumps({k: result[k] for k in ("kind", "step", "nfe", "fid", "num_generated")}), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
