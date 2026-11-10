#!/usr/bin/env python3
"""Encode CelebA64 once as posterior means or fixed posterior samples."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from local_paths import REPO_ROOT
from src.celeba_latent import (
    DEFAULT_POSTERIOR_SAMPLE_SEED,
    LATENT_CACHE_ENCODINGS,
    CelebALatentCodec,
    encode_loader_to_cache,
)
from src.utils import load_celeba_dataset


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "celeba64")
    parser.add_argument("--split", default="training")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--vae-model",
        default=str(REPO_ROOT / "data" / "models" / "sd-vae-ft-mse"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "celeba64_latents_sdvae_ft_mse.pt",
    )
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache-dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument(
        "--encoding",
        choices=LATENT_CACHE_ENCODINGS,
        default="posterior_mean",
        help=(
            "posterior_mean preserves the historical deterministic cache; "
            "posterior_sample stores one fixed VAE posterior draw per absolute image index "
            "without flip augmentation"
        ),
    )
    parser.add_argument(
        "--posterior-sample-seed",
        type=int,
        default=DEFAULT_POSTERIOR_SAMPLE_SEED,
        help="base seed folded into each absolute image index for posterior_sample",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.posterior_sample_seed < 0:
        parser.error("--posterior-sample-seed must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = load_celeba_dataset(
        args.data_root,
        split=args.split,
        start_index=args.start_index,
        limit=args.limit,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    codec = CelebALatentCodec.from_pretrained(
        args.vae_model,
        device=device,
        local_files_only=True,
    )
    cache_dtype = torch.float16 if args.cache_dtype == "float16" else torch.float32
    paths = getattr(dataset, "paths", ())
    metadata = {
        "dataset_root": str(args.data_root.resolve()),
        "split": args.split,
        "start_index": args.start_index,
        "requested_limit": args.limit,
        "first_image": str(paths[0]) if paths else None,
        "last_image": str(paths[-1]) if paths else None,
    }
    latents, saved_metadata = encode_loader_to_cache(
        codec,
        tqdm(loader, desc="encode CelebA latents"),
        args.output,
        overwrite=args.overwrite,
        cache_dtype=cache_dtype,
        metadata=metadata,
        encoding=args.encoding,
        posterior_sample_seed=args.posterior_sample_seed,
        posterior_sample_index_start=args.start_index,
    )
    print(
        f"saved {latents.shape[0]} latents {tuple(latents.shape[1:])} "
        f"to {args.output}; mean={float(latents.float().mean()):.6f} "
        f"std={float(latents.float().std()):.6f} metadata={saved_metadata}"
    )


if __name__ == "__main__":
    main()
