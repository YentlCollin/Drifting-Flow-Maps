#!/usr/bin/env python3
"""Standard pixel DDPM training on the existing official CelebA64 image split.

Uses the already installed lucidrains GaussianDiffusion implementation, with
epsilon prediction, a fixed 1000-step schedule, and independently respaced DDIM
evaluation. Existing drifting and historical x0-DDIM code is left untouched.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

# Must precede torch import and use separate MIOpen databases for DDP workers.
if os.environ.get("DRIFTMODEL_MIOPEN_CACHE_BASE"):
    cache = Path(os.environ["DRIFTMODEL_MIOPEN_CACHE_BASE"]) / os.environ.get("LOCAL_RANK", "0")
    for sub in (cache, cache / "userdb", cache / "kernels"):
        sub.mkdir(parents=True, exist_ok=True)
    os.environ.update(MIOPEN_CACHE_DIR=str(cache), MIOPEN_USER_DB_PATH=str(cache / "userdb"),
                      MIOPEN_CUSTOM_CACHE_DIR=str(cache / "kernels"))

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from denoising_diffusion_pytorch import GaussianDiffusion, Unet

ROOT = Path(__file__).resolve().parents[2]


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class OfficialImages(Dataset):
    def __init__(self, directory, expected_count=162770):
        self.paths = sorted(Path(directory).glob("*.png"))
        if len(self.paths) != expected_count:
            raise ValueError(f"expected {expected_count} official training images, got {len(self.paths)}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        if pixels.shape != (64, 64, 3):
            raise ValueError(f"expected existing canonical RGB64 preprocessing: {self.paths[index]}")
        return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255)


def build_diffusion(config):
    network = Unet(dim=config["dim"], dim_mults=tuple(config["dim_mults"]), channels=3,
                   dropout=config["dropout"], flash_attn=False)
    return GaussianDiffusion(network, image_size=64, timesteps=config["timesteps"],
                             objective="pred_noise", beta_schedule=config["beta_schedule"],
                             auto_normalize=True, min_snr_loss_weight=False)


def setup_device(device):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, world, rank, local_rank


def get_rng(device):
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng(value, device):
    torch.set_rng_state(value["torch"])
    if value["cuda"] is not None:
        torch.cuda.set_rng_state(value["cuda"], device)
    np.random.set_state(value["numpy"])
    random.setstate(value["python"])


@torch.no_grad()
def update_ema(ema, model, step, target_decay):
    decay = min(target_decay, (1 + step) / (10 + step))
    torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - decay)
    # Schedule buffers are constant, so only parameters need updating.


def save_checkpoint(out, raw, ema, optimizer, step, config, device, world, rank, milestone):
    local_rng = get_rng(device)
    rng = [None] * world if rank == 0 else None
    if world > 1:
        dist.gather_object(local_rng, rng, dst=0)
    else:
        rng = [local_rng]
    if rank == 0:
        payload = {"format_version": 1, "step": step, "config": config,
                   "model": raw.state_dict(), "ema": ema.state_dict(),
                   "optimizer": optimizer.state_dict(), "rng_by_rank": rng,
                   "world_size": world}
        temporary = out / "checkpoint_latest.pt.tmp"
        torch.save(payload, temporary)
        os.replace(temporary, out / "checkpoint_latest.pt")
        if milestone:
            # A hard link adds no second checkpoint-sized allocation on shared storage.
            path = out / f"checkpoint_{step:08d}.pt"
            if not path.exists():
                os.link(out / "checkpoint_latest.pt", path)
        atomic_json(out / "status.json", {"step": step, "checkpoint": str(out / "checkpoint_latest.pt"),
                                           "unix_time": time.time(), "job_id": os.environ.get("SLURM_JOB_ID")})
    if world > 1:
        dist.barrier()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/celeba64_official_bilinear/training")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--batch", type=int, default=32, help="per GPU batch size")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--dim-mults", type=int, nargs="+", default=[1, 2, 2, 2])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--timesteps", type=int, default=1000)
    p.add_argument("--beta-schedule", choices=["linear", "cosine", "sigmoid"], default="cosine")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--horizontal-flip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--checkpoint-every", type=int, default=10000)
    p.add_argument("--rolling-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--resume", type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--expected-count", type=int, default=162770, help="override only for explicit synthetic smoke tests")
    return p.parse_args()


def main():
    args = parse_args()
    device, world, rank, local_rank = setup_device(args.device)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    raw = build_diffusion(vars(args)).to(device)
    ema = copy.deepcopy(raw).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(raw.parameters(), lr=args.lr, betas=(0.9, 0.999))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(world_size=world, global_batch=args.batch * world, torch_version=torch.__version__,
                  parameters=sum(p.numel() for p in raw.parameters()),
                  prediction="epsilon", image_range="[0,1] input, [-1,1] diffusion", sampler="DDIM eta=0",
                  source_sha256=sha256(__file__), dataset="CelebA official training; existing crop178 bilinear64",
                  manifest_sha256=sha256(args.data_dir.parent / "manifest.json") if (args.data_dir.parent / "manifest.json").exists() else None)
    start_step = 0
    resume_rng = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        for key in ("dim", "dim_mults", "dropout", "timesteps", "beta_schedule", "batch", "seed", "lr",
                    "warmup", "ema_decay", "horizontal_flip", "precision", "data_dir", "grad_clip"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"resume changes scientific configuration: {key}")
        if checkpoint["world_size"] != world:
            raise ValueError("resume requires same world size for data and RNG continuation")
        raw.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        resume_rng = checkpoint["rng_by_rank"][rank]
        del checkpoint
    output_error = [None]
    if rank == 0 and not args.resume and args.out.exists() and any(args.out.iterdir()):
        output_error[0] = f"nonempty output requires --resume: {args.out}"
    if world > 1:
        dist.broadcast_object_list(output_error, src=0)
    if output_error[0]:
        raise FileExistsError(output_error[0])
    if args.steps <= start_step:
        raise ValueError("--steps must exceed the current checkpoint step")
    args.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        atomic_json(args.out / f"config_start{start_step:08d}.json", config)
        print(json.dumps({"event": "start", **config}), flush=True)
    model = DistributedDataParallel(raw, device_ids=[local_rank], broadcast_buffers=False) if world > 1 and device.type == "cuda" else (DistributedDataParallel(raw, broadcast_buffers=False) if world > 1 else raw)
    torch.manual_seed(args.seed + rank)
    dataset = OfficialImages(args.data_dir, args.expected_count)
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    # A separate generator keeps DataLoader iterator construction off the model RNG stream.
    loader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, drop_last=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda",
                        persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))
    if len(loader) == 0:
        raise ValueError("dataset too small for global batch")
    epoch, offset = divmod(start_step, len(loader))
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(offset):
        next(iterator)
    if resume_rng:
        set_rng(resume_rng, device)
    begin = time.monotonic()
    interval_begin = begin
    losses, norms = [], []
    raw.train()
    for step in range(start_step + 1, args.steps + 1):
        try:
            images = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            images = next(iterator)
        images = images.to(device, non_blocking=True)
        if args.horizontal_flip:
            flip = torch.rand(images.shape[0], device=device) < 0.5
            images = torch.where(flip[:, None, None, None], images.flip(-1), images)
        lr = args.lr * min(1., step / max(1, args.warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            loss = model(images)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        update_ema(ema, raw, step, args.ema_decay)
        losses.append(loss.detach())
        norms.append(norm.detach())
        if step % args.log_every == 0 or step == args.steps:
            stats = torch.stack([torch.stack(losses).mean(), torch.stack(norms).mean()])
            if world > 1:
                dist.all_reduce(stats)
                stats.div_(world)
            loss_value, norm_value = stats.cpu().tolist()
            now = time.monotonic()
            row = {"step": step, "loss": loss_value, "grad_norm": norm_value, "lr": lr,
                   "seconds_per_step": (now - interval_begin) / len(losses), "elapsed_seconds": now - begin,
                   "images_seen": step * args.batch * world,
                   "gpu_hours_this_session": (now - begin) * world / 3600 if device.type == "cuda" else 0,
                   "gpu_memory_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0}
            if rank == 0:
                print(json.dumps(row), flush=True)
                with (args.out / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
            losses, norms = [], []
            interval_begin = time.monotonic()
        milestone = step % args.checkpoint_every == 0 or step == args.steps
        if milestone or step % args.rolling_every == 0:
            save_checkpoint(args.out, raw, ema, optimizer, step, config, device, world, rank, milestone)
            interval_begin = time.monotonic()
    if rank == 0:
        atomic_json(args.out / f"completed_{args.steps:08d}.json", {"step": args.steps, "seconds": time.monotonic() - begin,
                                                                  "job_id": os.environ.get("SLURM_JOB_ID")})
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
