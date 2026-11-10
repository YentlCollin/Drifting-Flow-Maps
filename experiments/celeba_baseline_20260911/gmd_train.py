#!/usr/bin/env python3
"""Official CelebA training with the established MAE82 + ConvNeXt13 GMD loss.

The shared trainer owns data loading, optimization, checkpoint integrity and
exact resume. A short run uses --steps 20; resuming it to --steps 25000 keeps
the same objective and learning-rate schedule. This is a fresh official-split
experiment, not a continuation of the mixed-split FID-18.89 legacy screen.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

import celeba_pixel_gmd_generator_train as base
from src.celeba_pixel_gmd_convnext import load_local_convnextv2_base
from src.celeba_pixel_gmd_features import load_frozen_pixel_feature_encoder
from src.celeba_pixel_gmd_mae_convnext import (
    FrozenMAEConvNextFeatures,
    composite_representation_signature,
    global_mae_convnext_gmd_loss,
)

TRAIN_DIR = REPO_ROOT / "data/celeba64_official_bilinear/training"
TRAIN_FILES_SHA256 = "79b3854e02d591e25eede33f38690d16a1faa6aac57ca77c66844156975c2205"
MAE_PATH = REPO_ROOT / "runs/celeba_pixel_gmd/pixel_mae_official_train_base64_seed42_192ep_v1/feature_encoder_ema_final.pt"
MAE_SHA256 = "9d088740c9cf725809a3d80b124591dc2f229d3c7772b177257320e84bb7d985"
MAE_SIGNATURE_SHA256 = "fff1d679590be76f3b2089bda88d32c496f591dd27e41ae66451c7ebc230f621"
SOURCE_FILES = (
    'experiments/celeba_baseline_20260911/gmd_train.py',
    'celeba_pixel_gmd_generator_train.py',
    'celeba_pixel_gmd_mae_train.py',
    'src/celeba_pixel_gmd_generator.py',
    'src/celeba_pixel_gmd_features.py',
    'src/official_drifting.py',
    'src/celeba_pixel_gmd_convnext.py',
    'src/celeba_pixel_gmd_mae_convnext.py',
)


def build_parser() -> argparse.ArgumentParser:
    parser = base.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        data_dir=TRAIN_DIR,
        train_start_index=0,
        train_end_index=162770,
        feature_checkpoint=MAE_PATH,
        feature_checkpoint_sha256=MAE_SHA256,
        expected_feature_training_signature_sha256=MAE_SIGNATURE_SHA256,
        steps=25000,
        checkpoint_every=2500,
        lr_warmup_steps=10000,
        verify_gradients=True,
        run_label="official_mae82_convnext13_seed42_25k_20260911",
    )
    for action in parser._actions:
        if action.dest in {"feature_checkpoint", "feature_checkpoint_sha256"}:
            action.required = False
    parser.add_argument(
        "--convnext-snapshot-dir", type=Path,
        default=REPO_ROOT / "data/models/convnextv2-base-22k-224_758ff092",
    )
    return parser


def validate_official_inputs(args: argparse.Namespace) -> None:
    if args.smoke:
        raise ValueError("Use --steps 20 for a full-architecture GPU smoke; --smoke changes the scientific recipe")
    expected = {
        "train_start_index": 0,
        "train_end_index": 162770,
        "feature_checkpoint_sha256": MAE_SHA256,
        "expected_feature_training_signature_sha256": MAE_SIGNATURE_SHA256,
        "feature_weight_kind": "ema",
        "allow_short_feature_checkpoint_preflight": False,
    }
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f"official baseline requires {name}={value!r}")
    if args.data_dir.resolve() != TRAIN_DIR.resolve():
        raise ValueError("official baseline requires the official training directory")
    if args.feature_checkpoint.resolve() != MAE_PATH.resolve():
        raise ValueError("official baseline requires the official 192-epoch EMA MAE")


def install_hooks(args: argparse.Namespace) -> None:
    original_signature = base.training_signature
    original_verify = base.verify_gradients
    active: dict[str, FrozenMAEConvNextFeatures] = {}

    def load_composite(checkpoint, device="cpu"):
        mae = load_frozen_pixel_feature_encoder(checkpoint, device)
        convnext = load_local_convnextv2_base(args.convnext_snapshot_dir, device=device)
        composite = FrozenMAEConvNextFeatures(mae, convnext).eval()
        active["encoder"] = composite
        return composite

    def signature(signature_args, dataset_manifest, generator, feature_identity):
        if (dataset_manifest["files_sha256"] != TRAIN_FILES_SHA256
                or dataset_manifest["image_count"] != 162770):
            raise ValueError("training corpus differs from the pinned official split")
        if feature_identity["completed_step"] != 121920:
            raise ValueError("official MAE must have completed exactly 121920 updates")
        result = original_signature(signature_args, dataset_manifest, generator, feature_identity)
        result.update(
            objective="released_normalized_multitemperature_GMD_MAE82_plus_ConvNeXt13",
            objective_implementation="src.celeba_pixel_gmd_mae_convnext.global_mae_convnext_gmd_loss",
            feature_representation=composite_representation_signature(active["encoder"]),
            precision="FP32 params/optimizer/GMD; BF16 generator/MAE autocast; ConvNeXt FP32",
            experiment="official_celeba_mae82_convnext13_baseline_20260911",
        )
        result["support"]["feature_extraction_distribution"] = (
            "differentiable all_gather first; complete generated/positive/negative supports "
            "on every rank; concatenate ConvNeXt chunks 8/32 before per-family loss"
        )
        return result

    def verify_gradients(model, feature_encoder, verify_args):
        original_verify(model, feature_encoder, verify_args)
        pixels = feature_encoder.last_local_generated
        valid = pixels is not None and pixels.grad is not None
        if valid:
            valid = bool(torch.isfinite(pixels.grad).all()) and bool(pixels.grad.abs().sum() > 0)
        passed = torch.tensor(int(valid), dtype=torch.int32, device=verify_args.device)
        if verify_args.distributed:
            dist.all_reduce(passed, op=dist.ReduceOp.MIN)
        if not bool(passed):
            raise FloatingPointError("generated RGB gradient must be finite and nonzero on every rank")
        feature_encoder.clear_step_state()

    base.SOURCE_FILES = SOURCE_FILES
    base.load_frozen_pixel_feature_encoder = load_composite
    base.global_pixel_gmd_loss = global_mae_convnext_gmd_loss
    base.training_signature = signature
    base.verify_gradients = verify_gradients


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    args.command = shlex.join([sys.argv[0], *(sys.argv[1:] if argv is None else argv)])
    validate_official_inputs(args)
    install_hooks(args)
    try:
        base.setup_distributed(args)
        return base.train(args)
    finally:
        base.cleanup_distributed(args)


if __name__ == "__main__":
    main()
